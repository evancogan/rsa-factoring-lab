// sieve4.cu - one more optimization round on the sieving itself.
//
// FIX 1 - THE CROWDED BANK (powers of 2).
//   Shared memory is split into 32 "banks"; number i lives in bank i % 32. When the whole block
//   walks prime 2, thread t hits number 2t, so pairs of threads land in the same bank and must
//   take turns. For 4 it's groups of 4, for 32 all 32 threads queue at ONE bank.
//   But we don't need walkers for 2 at all: in binary, "how many times does 2 divide n?" is just
//   the number of trailing zero bits, which the GPU counts in a single instruction. So every
//   tally now STARTS at 4 x trailing_zeros(n), and the 2-walkers are gone. Every walker left
//   has an odd stride, and odd strides spread across all 32 banks with no queueing.
//
// FIX 2 - WALKERS REMEMBER WHERE THEY ARE.
//   sieve3 recomputed every walker's first stop in every segment with a 64-bit division
//   (base % q), 847 of them per segment - division again, the GPU's weak spot. Now each block
//   sieves a contiguous run of segments, and each walker just carries its position forward:
//   next_offset = offset - (SEG mod q), wrapped around. One subtraction instead of a division.
//
// The CPU gets both fixes too, so the race stays fair.
//
// Usage:  ./sieve4  |  ./sieve4 500000000  |  ./sieve4 2000000000
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

constexpr uint64_t START = 1000000000000ULL;
constexpr uint32_t BOUND = 1000;
constexpr int SEG      = 8192;
constexpr int BLOCK    = 256;
constexpr int SMALL    = 256;
constexpr int CAND_MAX = 1024;
constexpr int OWN      = 4;      // prime powers owned per thread (supports up to 1024)
constexpr int CPU_SEG  = 32768;

__constant__ uint64_t c_inv[167];
__constant__ uint64_t c_lim[167];

__host__ __device__ inline int tz(uint64_t n) {          // trailing zero bits of n
#ifdef __CUDA_ARCH__
    return __ffsll((long long)n) - 1;
#else
    return __builtin_ctzll(n);
#endif
}

__host__ __device__ inline bool smooth_fast(uint64_t n, const uint64_t* inv, const uint64_t* lim, int nodd) {
    n >>= tz(n);
    for (int i = 0; i < nodd; i++) {
        uint64_t m;
        while ((m = n * inv[i]) <= lim[i]) n = m;
        if (n < BOUND) return true;
    }
    return n < BOUND;
}

__global__ void sieve_kernel(uint64_t start, uint64_t count, const uint64_t* pp, const uint32_t* segmod,
                             const uint8_t* lg, int npp, int nsmall, int nodd, uint32_t thresh,
                             unsigned long long* total) {
    __shared__ uint32_t logs[SEG];
    __shared__ uint32_t offs[1024];
    __shared__ uint64_t cand[CAND_MAX];
    __shared__ unsigned ncand, block_count;
    if (threadIdx.x == 0) block_count = 0;

    // FIX 2: this block sieves segments s0..s1-1 in order.
    uint64_t nseg = (count + SEG - 1) / SEG;
    uint64_t per  = (nseg + gridDim.x - 1) / gridDim.x;
    uint64_t s0 = (uint64_t)blockIdx.x * per;
    uint64_t s1 = s0 + per < nseg ? s0 + per : nseg;

    // Each thread owns up to OWN walkers and keeps their positions in registers.
    uint64_t my_off[OWN], my_q[OWN];
    uint32_t my_sm[OWN];
    #pragma unroll
    for (int j = 0; j < OWN; j++) {
        int k = threadIdx.x + j * blockDim.x;
        if (k < npp) {
            uint64_t q = pp[k], b = start + s0 * SEG;
            my_q[j] = q; my_sm[j] = segmod[k];
            my_off[j] = (q - b % q) % q;           // the only division, once per walker per block
        }
    }

    for (uint64_t s = s0; s < s1; s++) {
        uint64_t base = start + s * SEG;
        if (threadIdx.x == 0) ncand = 0;
        // FIX 1: start each tally with the powers of 2, straight from the bits.
        for (int i = threadIdx.x; i < SEG; i += blockDim.x) logs[i] = 4 * tz(base + i);
        #pragma unroll
        for (int j = 0; j < OWN; j++) {
            int k = threadIdx.x + j * blockDim.x;
            if (k < npp) {
                offs[k] = my_off[j] < SEG ? (uint32_t)my_off[j] : SEG;
                my_off[j] = my_off[j] >= my_sm[j] ? my_off[j] - my_sm[j]           // step to the
                                                  : my_off[j] + my_q[j] - my_sm[j]; // next segment
            }
        }
        __syncthreads();

        for (int k = 0; k < nsmall; k++) {              // small odd walkers: whole block shares
            uint32_t q = (uint32_t)pp[k], l = lg[k];
            for (uint32_t i = offs[k] + threadIdx.x * q; i < SEG; i += blockDim.x * q)
                atomicAdd(&logs[i], l);
        }
        for (int k = nsmall + threadIdx.x; k < npp; k += blockDim.x) {   // large: one thread each
            uint64_t q = pp[k]; uint32_t l = lg[k];
            for (uint64_t i = offs[k]; i < SEG; i += q) atomicAdd(&logs[i], l);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < SEG; i += blockDim.x)
            if (base + i < start + count && logs[i] >= thresh) {
                unsigned idx = atomicAdd(&ncand, 1u);
                if (idx < CAND_MAX) cand[idx] = base + i;
                else if (smooth_fast(base + i, c_inv, c_lim, nodd)) atomicAdd(&block_count, 1u);
            }
        __syncthreads();

        unsigned nc = min(ncand, (unsigned)CAND_MAX);
        for (unsigned j = threadIdx.x; j < nc; j += blockDim.x)
            if (smooth_fast(cand[j], c_inv, c_lim, nodd)) atomicAdd(&block_count, 1u);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, (unsigned long long)block_count);
}

static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    uint64_t COUNT = argc > 1 ? strtoull(argv[1], nullptr, 10) : 50000000ULL;

    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p < BOUND; p++) {
        bool prime = true;
        for (uint32_t q = 2; q * q <= p; q++) if (p % q == 0) { prime = false; break; }
        if (prime) primes.push_back(p);
    }
    std::vector<uint64_t> inv, lim;
    for (size_t k = 1; k < primes.size(); k++) {
        uint64_t p = primes[k], x = p;
        for (int i = 0; i < 6; i++) x *= 2 - p * x;
        inv.push_back(x); lim.push_back(UINT64_MAX / p);
    }
    int nodd = (int)inv.size();

    // Walkers for ODD prime powers only - powers of 2 are handled by counting trailing zeros.
    uint64_t maxN = START + COUNT - 1;
    std::vector<std::pair<uint64_t, uint8_t>> v;
    for (size_t idx = 1; idx < primes.size(); idx++) {
        uint32_t p = primes[idx];
        uint8_t l = (uint8_t)lround(4 * log2((double)p));
        for (uint64_t q = p;; q *= p) { v.push_back({q, l}); if (q > maxN / p) break; }
    }
    std::sort(v.begin(), v.end());
    int npp = (int)v.size(), nsmall = 0;
    while (nsmall < npp && v[nsmall].first < SMALL) nsmall++;
    if (npp > OWN * BLOCK) { printf("Too many prime powers for OWN\n"); return 1; }
    std::vector<uint64_t> pp(npp); std::vector<uint8_t> lg(npp); std::vector<uint32_t> segmod(npp);
    for (int k = 0; k < npp; k++) {
        pp[k] = v[k].first; lg[k] = v[k].second;
        segmod[k] = (uint32_t)(pp[k] > (uint64_t)SEG ? SEG : SEG % pp[k]);
    }
    uint32_t thresh = (uint32_t)floor(4 * log2((double)START)) - 16;

    printf("Sieving %llu numbers starting at %llu with %d odd prime powers (+ powers of 2 via bit-counting)\n\n",
           (unsigned long long)COUNT, (unsigned long long)START, npp);

    // ---- CPU sieve: same two fixes (contiguous chunks + walkers keep their place) ----
    auto t0 = std::chrono::steady_clock::now();
    unsigned long long cpu = 0;
    uint64_t nseg = (COUNT + CPU_SEG - 1) / CPU_SEG;
    #pragma omp parallel reduction(+:cpu)
    {
        uint64_t T = omp_get_num_threads(), t = omp_get_thread_num();
        uint64_t per = (nseg + T - 1) / T, s0 = t * per, s1 = std::min(s0 + per, nseg);
        std::vector<uint8_t> logs(CPU_SEG);
        std::vector<uint64_t> off(npp);
        uint64_t b0 = START + s0 * CPU_SEG;
        for (int k = 0; k < npp; k++) off[k] = (pp[k] - b0 % pp[k]) % pp[k];
        for (uint64_t s = s0; s < s1; s++) {
            uint64_t base = START + s * CPU_SEG;
            for (int i = 0; i < CPU_SEG; i++) logs[i] = (uint8_t)(4 * tz(base + i));
            for (int k = 0; k < npp; k++) {
                uint64_t q = pp[k], i = off[k];
                for (; i < (uint64_t)CPU_SEG; i += q) logs[i] += lg[k];
                off[k] = i - CPU_SEG;                  // carry position into the next segment
            }
            for (int i = 0; i < CPU_SEG; i++)
                if (base + i < START + COUNT && logs[i] >= thresh &&
                    smooth_fast(base + i, inv.data(), lim.data(), nodd)) cpu++;
        }
    }
    double cpu_s = since(t0);
    printf("CPU sieve (%2d threads): %llu smooth numbers in %.3f s\n", omp_get_max_threads(), cpu, cpu_s);

    // ---- GPU ----
    CHECK(cudaFree(0));
    uint64_t* d_pp; uint32_t* d_sm; uint8_t* d_lg; unsigned long long* d_total;
    CHECK(cudaMalloc(&d_pp, npp * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_sm, npp * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_lg, npp));
    CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_pp, pp.data(), npp * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sm, segmod.data(), npp * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lg, lg.data(), npp, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_inv, inv.data(), nodd * sizeof(uint64_t)));
    CHECK(cudaMemcpyToSymbol(c_lim, lim.data(), nodd * sizeof(uint64_t)));
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int blocks = sms * 8;

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));   // warm-up
    sieve_kernel<<<blocks, BLOCK>>>(START, SEG, d_pp, d_sm, d_lg, npp, nsmall, nodd, thresh, d_total);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    auto t1 = std::chrono::steady_clock::now();
    sieve_kernel<<<blocks, BLOCK>>>(START, COUNT, d_pp, d_sm, d_lg, npp, nsmall, nodd, thresh, d_total);
    CHECK(cudaGetLastError());
    unsigned long long gpu;
    CHECK(cudaMemcpy(&gpu, d_total, sizeof(gpu), cudaMemcpyDeviceToHost));
    double gpu_s = since(t1);

    printf("GPU sieve (%d threads): %llu smooth numbers in %.3f s  (everything on the GPU)\n\n",
           blocks * BLOCK, gpu, gpu_s);
    printf("Answers %s.  GPU speedup: %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    printf("sieve3 on your PC was: CPU 0.347 s, GPU 0.149 s at 2 billion\n");
    return 0;
}
