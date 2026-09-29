// sieve2.cu - Goal 1: the GPU checks its own candidates.
// In sieve.cu the GPU sieved in 3 ms, then sat idle while 211,156 candidates were copied to the
// CPU and checked there. Here each GPU block confirms its own candidates on the spot, so the only
// thing that crosses back to the CPU is one final number. "Make GPU never block on CPU."
//
// Usage:  ./sieve2              (50 million numbers, same as before)
//         ./sieve2 2000000000   (2 billion numbers - see what happens when the job is big)
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

const uint64_t START = 1000000000000ULL;
const uint32_t BOUND = 1000;
const int SEG   = 8192;
const int BLOCK = 256;
const int SMALL = 256;
const int CPU_SEG = 32768;

__constant__ uint32_t c_primes[168];   // the small primes, for the final exact check on the GPU

static bool is_smooth(uint64_t n, const std::vector<uint32_t>& primes) {
    for (uint32_t p : primes) { while (n % p == 0) n /= p; if (n == 1) return true; }
    return n == 1;
}

__device__ bool is_smooth_gpu(uint64_t n, int np) {
    for (int i = 0; i < np; i++) {
        uint32_t p = c_primes[i];
        while (n % p == 0) n /= p;
        if (n == 1) return true;
    }
    return n == 1;
}

__global__ void sieve_kernel(uint64_t start, uint64_t count, const uint64_t* pp, const uint8_t* lg,
                             int npp, int nsmall, int np, uint32_t thresh,
                             unsigned long long* total) {
    __shared__ uint32_t logs[SEG];
    __shared__ uint32_t offs[1024];
    __shared__ unsigned block_count;          // this block's running tally of confirmed smooth numbers
    if (threadIdx.x == 0) block_count = 0;
    uint64_t nseg = (count + SEG - 1) / SEG;

    for (uint64_t s = blockIdx.x; s < nseg; s += gridDim.x) {
        uint64_t base = start + s * SEG;
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

        // NEW: confirm candidates right here instead of shipping them to the CPU.
        for (int i = threadIdx.x; i < SEG; i += blockDim.x)
            if (base + i < start + count && logs[i] >= thresh && is_smooth_gpu(base + i, np))
                atomicAdd(&block_count, 1u);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, (unsigned long long)block_count);  // one number per block
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
    int np = (int)primes.size();
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

    // ---- CPU sieve (unchanged from sieve.cu) ----
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
                if (base + i < START + COUNT && logs[i] >= thresh && is_smooth(base + i, primes)) cpu++;
        }
    }
    double cpu_s = since(t0);
    printf("CPU sieve (%2d threads): %llu smooth numbers in %.3f s\n", omp_get_max_threads(), cpu, cpu_s);

    // ---- GPU sieve + GPU check ----
    CHECK(cudaFree(0));
    uint64_t* d_pp; uint8_t* d_lg; unsigned long long* d_total;
    CHECK(cudaMalloc(&d_pp, npp * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_lg, npp));
    CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_pp, pp.data(), npp * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lg, lg.data(), npp, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_primes, primes.data(), np * sizeof(uint32_t)));
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int blocks = sms * 8;

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));   // warm-up
    sieve_kernel<<<blocks, BLOCK>>>(START, SEG, d_pp, d_lg, npp, nsmall, np, thresh, d_total);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    auto t1 = std::chrono::steady_clock::now();
    sieve_kernel<<<blocks, BLOCK>>>(START, COUNT, d_pp, d_lg, npp, nsmall, np, thresh, d_total);
    CHECK(cudaGetLastError());
    unsigned long long gpu;
    CHECK(cudaMemcpy(&gpu, d_total, sizeof(gpu), cudaMemcpyDeviceToHost));   // the only thing copied back
    double gpu_s = since(t1);

    printf("GPU sieve (%d threads): %llu smooth numbers in %.3f s  (everything on the GPU)\n\n",
           blocks * BLOCK, gpu, gpu_s);
    printf("Answers %s.  GPU speedup: %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    return 0;
}
