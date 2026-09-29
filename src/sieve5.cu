// sieve5.cu - optimizations chosen by the PROFILER, not by guessing.
//
// Nsight Compute said about sieve4:
//   (a) Occupancy 33%: each block needed 44 KB of shared memory, so only 2 blocks fit per SM.
//       Est. speedup up to 67%.
//   (b) 40% of stall time: warps waiting at __syncthreads() for slower sibling warps.
//       Est. speedup up to 40%.
//   (c) Only ~18 of 32 threads per warp active on average (divergence). Est. up to 12%.
//
// FIX A - SMALLER FOOTPRINT: every tally stays below 256, so it fits in ONE BYTE. Pack 4 tallies
//         into each 32-bit word and add with a shifted atomicAdd (a byte never overflows into its
//         neighbour). The tally array shrinks from 32 KB to 8 KB, and the candidate list from
//         8 KB to 4 KB. About 16 KB per block, so several more blocks fit on each SM.
// FIX B - ONE FULL WAVE: ask CUDA how many blocks fit per SM and launch exactly that many, so no
//         half-empty second "wave" of blocks runs at the end.
// FIX C - WHOLE WARPS CHECK CANDIDATES: before, ~35 candidates were checked by ~35 threads while
//         the other ~220 threads sat at the barrier. Now each candidate is checked by a whole warp:
//         the 32 threads split the 167 odd primes between them, each strips out its own primes,
//         and a warp "shuffle" multiplies their findings together. If the product equals the
//         number, it was built entirely from small primes. All 8 warps stay busy.
//
// CPU: back to sieve3's on-demand work sharing (better for your P-core/E-core mix), plus the
//      trailing-zeros trick for powers of 2.
//
// Usage:  ./sieve5  |  ./sieve5 500000000  |  ./sieve5 2000000000
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
constexpr int CAND_MAX = 512;
constexpr int OWN      = 4;
constexpr int CPU_SEG  = 32768;
constexpr int NODD     = 167;    // odd primes below 1000

__constant__ uint64_t c_inv[NODD];
__constant__ uint64_t c_lim[NODD];
__constant__ uint32_t c_podd[NODD];

__host__ __device__ inline int tz(uint64_t n) {
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

// FIX C: all 32 lanes of a warp call this with the same n.
__device__ bool smooth_warp(uint64_t n, int lane) {
    uint64_t odd = n >> tz(n);
    uint64_t found = 1;                               // product of this lane's prime powers
    for (int i = lane; i < NODD; i += 32) {
        uint64_t x = odd, y;
        while ((y = x * c_inv[i]) <= c_lim[i]) { x = y; found *= c_podd[i]; }
    }
    for (int o = 16; o > 0; o >>= 1)                  // multiply all 32 lanes' results together
        found *= __shfl_xor_sync(0xffffffffu, found, o);
    return found == odd;
}

__global__ void sieve_kernel(uint64_t start, uint64_t count, const uint64_t* pp, const uint32_t* segmod,
                             const uint8_t* lg, int npp, int nsmall, uint32_t thresh,
                             unsigned long long* total) {
    __shared__ uint32_t words[SEG / 4];   // FIX A: 4 one-byte tallies per word (8 KB)
    __shared__ uint32_t offs[1024];
    __shared__ uint64_t cand[CAND_MAX];
    __shared__ unsigned ncand, block_count;
    if (threadIdx.x == 0) block_count = 0;

    uint64_t nseg = (count + SEG - 1) / SEG;
    uint64_t per  = (nseg + gridDim.x - 1) / gridDim.x;
    uint64_t s0 = (uint64_t)blockIdx.x * per;
    uint64_t s1 = s0 + per < nseg ? s0 + per : nseg;

    uint64_t my_off[OWN], my_q[OWN];
    uint32_t my_sm[OWN];
    #pragma unroll
    for (int j = 0; j < OWN; j++) {
        int k = threadIdx.x + j * blockDim.x;
        if (k < npp) {
            uint64_t q = pp[k], b = start + s0 * SEG;
            my_q[j] = q; my_sm[j] = segmod[k];
            my_off[j] = (q - b % q) % q;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;

    for (uint64_t s = s0; s < s1; s++) {
        uint64_t base = start + s * SEG;
        if (threadIdx.x == 0) ncand = 0;
        for (int w = threadIdx.x; w < SEG / 4; w += blockDim.x) {       // start with powers of 2
            uint64_t n = base + 4 * (uint64_t)w;
            words[w] =  (uint32_t)(4 * tz(n))            | (uint32_t)(4 * tz(n + 1)) << 8
                     | (uint32_t)(4 * tz(n + 2)) << 16   | (uint32_t)(4 * tz(n + 3)) << 24;
        }
        #pragma unroll
        for (int j = 0; j < OWN; j++) {
            int k = threadIdx.x + j * blockDim.x;
            if (k < npp) {
                offs[k] = my_off[j] < SEG ? (uint32_t)my_off[j] : SEG;
                my_off[j] = my_off[j] >= my_sm[j] ? my_off[j] - my_sm[j] : my_off[j] + my_q[j] - my_sm[j];
            }
        }
        __syncthreads();

        for (int k = 0; k < nsmall; k++) {
            uint32_t q = (uint32_t)pp[k], l = lg[k];
            for (uint32_t i = offs[k] + threadIdx.x * q; i < SEG; i += blockDim.x * q)
                atomicAdd(&words[i >> 2], l << ((i & 3) * 8));
        }
        for (int k = nsmall + threadIdx.x; k < npp; k += blockDim.x) {
            uint64_t q = pp[k]; uint32_t l = lg[k];
            for (uint64_t i = offs[k]; i < SEG; i += q)
                atomicAdd(&words[i >> 2], l << ((i & 3) * 8));
        }
        __syncthreads();

        for (int i = threadIdx.x; i < SEG; i += blockDim.x) {
            uint32_t tally = (words[i >> 2] >> ((i & 3) * 8)) & 0xFF;
            if (base + i < start + count && tally >= thresh) {
                unsigned idx = atomicAdd(&ncand, 1u);
                if (idx < CAND_MAX) cand[idx] = base + i;
                else if (smooth_fast(base + i, c_inv, c_lim, NODD)) atomicAdd(&block_count, 1u);
            }
        }
        __syncthreads();

        unsigned nc = min(ncand, (unsigned)CAND_MAX);
        for (unsigned j = warp; j < nc; j += nwarps)                      // FIX C
            if (smooth_warp(cand[j], lane) && lane == 0) atomicAdd(&block_count, 1u);
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
    std::vector<uint64_t> inv, lim; std::vector<uint32_t> podd;
    for (size_t k = 1; k < primes.size(); k++) {
        uint64_t p = primes[k], x = p;
        for (int i = 0; i < 6; i++) x *= 2 - p * x;
        inv.push_back(x); lim.push_back(UINT64_MAX / p); podd.push_back((uint32_t)p);
    }
    if ((int)inv.size() != NODD) { printf("Expected %d odd primes\n", NODD); return 1; }

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

    printf("Sieving %llu numbers starting at %llu\n\n", (unsigned long long)COUNT, (unsigned long long)START);

    // ---- CPU: sieve3-style on-demand chunks + powers of 2 from trailing zeros ----
    auto t0 = std::chrono::steady_clock::now();
    unsigned long long cpu = 0;
    uint64_t nseg = (COUNT + CPU_SEG - 1) / CPU_SEG;
    #pragma omp parallel reduction(+:cpu)
    {
        std::vector<uint8_t> logs(CPU_SEG);
        #pragma omp for schedule(dynamic)
        for (long long s = 0; s < (long long)nseg; s++) {
            uint64_t base = START + (uint64_t)s * CPU_SEG;
            for (int i = 0; i < CPU_SEG; i++) logs[i] = (uint8_t)(4 * tz(base + i));
            for (int k = 0; k < npp; k++) {
                uint64_t q = pp[k];
                for (uint64_t i = (q - base % q) % q; i < (uint64_t)CPU_SEG; i += q) logs[i] += lg[k];
            }
            for (int i = 0; i < CPU_SEG; i++)
                if (base + i < START + COUNT && logs[i] >= thresh &&
                    smooth_fast(base + i, inv.data(), lim.data(), NODD)) cpu++;
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
    CHECK(cudaMemcpyToSymbol(c_inv, inv.data(), NODD * sizeof(uint64_t)));
    CHECK(cudaMemcpyToSymbol(c_lim, lim.data(), NODD * sizeof(uint64_t)));
    CHECK(cudaMemcpyToSymbol(c_podd, podd.data(), NODD * sizeof(uint32_t)));
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int per_sm; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sieve_kernel, BLOCK, 0));
    int blocks = sms * per_sm;                                        // FIX B: exactly one wave

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));       // warm-up
    sieve_kernel<<<blocks, BLOCK>>>(START, SEG, d_pp, d_sm, d_lg, npp, nsmall, thresh, d_total);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    auto t1 = std::chrono::steady_clock::now();
    sieve_kernel<<<blocks, BLOCK>>>(START, COUNT, d_pp, d_sm, d_lg, npp, nsmall, thresh, d_total);
    CHECK(cudaGetLastError());
    unsigned long long gpu;
    CHECK(cudaMemcpy(&gpu, d_total, sizeof(gpu), cudaMemcpyDeviceToHost));
    double gpu_s = since(t1);

    printf("GPU sieve (%d blocks = %d per SM x %d SMs): %llu smooth numbers in %.3f s\n\n",
           blocks, per_sm, sms, gpu, gpu_s);
    printf("Answers %s.  GPU speedup: %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    printf("Earlier at 2 billion: sieve3 CPU 0.347 s / GPU 0.149 s, sieve4 CPU 0.419 s / GPU 0.144 s\n");
    return 0;
}
