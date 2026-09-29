// sieve3.cu - fixing the new bottleneck.
// sieve2 moved the candidate check onto the GPU, but the check used 64-bit division, which is
// the GPU's weakest skill, and one candidate made its whole 32-thread warp wait. Two fixes:
//
// 1. DIVISION-FREE CHECK: for an odd prime p, precompute inv = p^-1 (mod 2^64). Then
//    "n is divisible by p"  <=>  n*inv <= (2^64-1)/p,  and when it is, n/p == n*inv exactly.
//    One multiply + one compare instead of a division. (Powers of 2 are just a bit shift.)
//    Early exit: once what's left of n is below 1000, all its prime factors are below 1000.
// 2. CANDIDATE COMPACTION: each block first gathers its candidates into a short list, then the
//    threads work through that list together, so busy threads aren't scattered across warps.
//
// The CPU gets fix #1 too, so the race stays fair.
//
// Usage:  ./sieve3  |  ./sieve3 500000000  |  ./sieve3 2000000000
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
constexpr int CPU_SEG  = 32768;

__constant__ uint64_t c_inv[167];   // inverses of the 167 odd primes below 1000
__constant__ uint64_t c_lim[167];

__host__ __device__ inline bool smooth_fast(uint64_t n, const uint64_t* inv, const uint64_t* lim, int nodd) {
#ifdef __CUDA_ARCH__
    n >>= (__ffsll((long long)n) - 1);      // strip all factors of 2
#else
    n >>= __builtin_ctzll(n);
#endif
    for (int i = 0; i < nodd; i++) {
        uint64_t m;
        while ((m = n * inv[i]) <= lim[i]) n = m;   // divisible -> m is the exact quotient
        if (n < BOUND) return true;
    }
    return n < BOUND;
}

__global__ void sieve_kernel(uint64_t start, uint64_t count, const uint64_t* pp, const uint8_t* lg,
                             int npp, int nsmall, int nodd, uint32_t thresh,
                             unsigned long long* total) {
    __shared__ uint32_t logs[SEG];
    __shared__ uint32_t offs[1024];
    __shared__ uint64_t cand[CAND_MAX];     // NEW: this segment's candidates, packed together
    __shared__ unsigned ncand, block_count;
    if (threadIdx.x == 0) block_count = 0;
    uint64_t nseg = (count + SEG - 1) / SEG;

    for (uint64_t s = blockIdx.x; s < nseg; s += gridDim.x) {
        uint64_t base = start + s * SEG;
        if (threadIdx.x == 0) ncand = 0;
        for (int i = threadIdx.x; i < SEG; i += blockDim.x) logs[i] = 0;
        for (int k = threadIdx.x; k < npp; k += blockDim.x) {
            uint64_t q = pp[k], off = (q - base % q) % q;
            offs[k] = off < SEG ? (uint32_t)off : SEG;
        }
        __syncthreads();

        for (int k = 0; k < nsmall; k++) {
            uint32_t q = (uint32_t)pp[k], l = lg[k];
            for (uint32_t i = offs[k] + threadIdx.x * q; i < SEG; i += blockDim.x * q)
                atomicAdd(&logs[i], l);
        }
        for (int k = nsmall + threadIdx.x; k < npp; k += blockDim.x) {
            uint64_t q = pp[k]; uint32_t l = lg[k];
            for (uint64_t i = offs[k]; i < SEG; i += q) atomicAdd(&logs[i], l);
        }
        __syncthreads();

        // Gather candidates into the packed list.
        for (int i = threadIdx.x; i < SEG; i += blockDim.x)
            if (base + i < start + count && logs[i] >= thresh) {
                unsigned idx = atomicAdd(&ncand, 1u);
                if (idx < CAND_MAX) cand[idx] = base + i;
                else if (smooth_fast(base + i, c_inv, c_lim, nodd)) atomicAdd(&block_count, 1u);  // rare overflow
            }
        __syncthreads();

        // Check the packed list: neighbouring threads get neighbouring candidates.
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
    // Inverses mod 2^64 for the odd primes (Newton's method: each step doubles the correct bits).
    std::vector<uint64_t> inv, lim;
    for (size_t k = 1; k < primes.size(); k++) {
        uint64_t p = primes[k], x = p;
        for (int i = 0; i < 6; i++) x *= 2 - p * x;
        inv.push_back(x); lim.push_back(UINT64_MAX / p);
    }
    int nodd = (int)inv.size();

    uint64_t maxN = START + COUNT - 1;
    std::vector<std::pair<uint64_t, uint8_t>> v;
    for (uint32_t p : primes) {
        uint8_t l = (uint8_t)lround(4 * log2((double)p));
        for (uint64_t q = p;; q *= p) { v.push_back({q, l}); if (q > maxN / p) break; }
    }
    std::sort(v.begin(), v.end());
    int npp = (int)v.size(), nsmall = 0;
    while (nsmall < npp && v[nsmall].first < SMALL) nsmall++;
    std::vector<uint64_t> pp(npp); std::vector<uint8_t> lg(npp);
    for (int k = 0; k < npp; k++) { pp[k] = v[k].first; lg[k] = v[k].second; }
    uint32_t thresh = (uint32_t)floor(4 * log2((double)START)) - 16;

    printf("Sieving %llu numbers starting at %llu with %d prime powers\n\n",
           (unsigned long long)COUNT, (unsigned long long)START, npp);

    // ---- CPU sieve (now with the division-free check too) ----
    auto t0 = std::chrono::steady_clock::now();
    unsigned long long cpu = 0;
    uint64_t nseg = (COUNT + CPU_SEG - 1) / CPU_SEG;
    #pragma omp parallel reduction(+:cpu)
    {
        std::vector<uint8_t> logs(CPU_SEG);
        #pragma omp for schedule(dynamic)
        for (long long s = 0; s < (long long)nseg; s++) {
            uint64_t base = START + (uint64_t)s * CPU_SEG;
            std::fill(logs.begin(), logs.end(), 0);
            for (int k = 0; k < npp; k++) {
                uint64_t q = pp[k];
                for (uint64_t i = (q - base % q) % q; i < (uint64_t)CPU_SEG; i += q) logs[i] += lg[k];
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
    uint64_t* d_pp; uint8_t* d_lg; unsigned long long* d_total;
    CHECK(cudaMalloc(&d_pp, npp * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_lg, npp));
    CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_pp, pp.data(), npp * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lg, lg.data(), npp, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_inv, inv.data(), nodd * sizeof(uint64_t)));
    CHECK(cudaMemcpyToSymbol(c_lim, lim.data(), nodd * sizeof(uint64_t)));
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int blocks = sms * 8;

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));   // warm-up
    sieve_kernel<<<blocks, BLOCK>>>(START, SEG, d_pp, d_lg, npp, nsmall, nodd, thresh, d_total);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    auto t1 = std::chrono::steady_clock::now();
    sieve_kernel<<<blocks, BLOCK>>>(START, COUNT, d_pp, d_lg, npp, nsmall, nodd, thresh, d_total);
    CHECK(cudaGetLastError());
    unsigned long long gpu;
    CHECK(cudaMemcpy(&gpu, d_total, sizeof(gpu), cudaMemcpyDeviceToHost));
    double gpu_s = since(t1);

    printf("GPU sieve (%d threads): %llu smooth numbers in %.3f s  (everything on the GPU)\n\n",
           blocks * BLOCK, gpu, gpu_s);
    printf("Answers %s.  GPU speedup: %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    return 0;
}
