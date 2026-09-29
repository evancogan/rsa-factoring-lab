// sieve.cu - same question as smooth.cu ("which numbers are built only from primes < 1000?"),
// but answered the way real factoring programs do it: SIEVING instead of dividing.
//
// Idea: the numbers divisible by 7 are exactly every 7th number. So instead of asking every
// number "are you divisible by 7?", walk down the line in steps of 7 and add log(7) to each
// stop. After all primes have walked, a number whose collected logs add up to its own size
// is fully built from small primes. No division needed for the heavy lifting.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

const uint64_t START = 1000000000000ULL;  // same range as smooth.cu
const uint64_t COUNT = 50000000ULL;
const uint32_t BOUND = 1000;
const int SEG   = 8192;   // GPU: numbers handled per block at a time (fits in fast shared memory)
const int BLOCK = 256;    // GPU: threads per block
const int SMALL = 256;    // prime powers below this are shared by the whole block (they have the most stops)
const int CPU_SEG = 32768;
const unsigned CAP = 4000000;

// Exact check, used only on the few numbers the sieve flags.
static bool is_smooth(uint64_t n, const std::vector<uint32_t>& primes) {
    for (uint32_t p : primes) { while (n % p == 0) n /= p; if (n == 1) return true; }
    return n == 1;
}

__global__ void sieve_kernel(uint64_t start, uint64_t count, const uint64_t* pp, const uint8_t* lg,
                             int npp, int nsmall, uint32_t thresh,
                             uint64_t* cand, unsigned* ncand, unsigned cap) {
    __shared__ uint32_t logs[SEG];    // one tally per number in this segment
    __shared__ uint32_t offs[1024];   // where each prime power's first stop lands in this segment
    uint64_t nseg = (count + SEG - 1) / SEG;

    for (uint64_t s = blockIdx.x; s < nseg; s += gridDim.x) {
        uint64_t base = start + s * SEG;
        for (int i = threadIdx.x; i < SEG; i += blockDim.x) logs[i] = 0;
        for (int k = threadIdx.x; k < npp; k += blockDim.x) {
            uint64_t q = pp[k], off = (q - base % q) % q;
            offs[k] = off < SEG ? (uint32_t)off : SEG;
        }
        __syncthreads();

        // Small prime powers (lots of stops): all 256 threads split the walk.
        for (int k = 0; k < nsmall; k++) {
            uint32_t q = (uint32_t)pp[k], l = lg[k];
            for (uint32_t i = offs[k] + threadIdx.x * q; i < SEG; i += blockDim.x * q)
                atomicAdd(&logs[i], l);
        }
        // Large prime powers (few stops): one thread walks each.
        for (int k = nsmall + threadIdx.x; k < npp; k += blockDim.x) {
            uint64_t q = pp[k]; uint32_t l = lg[k];
            for (uint64_t i = offs[k]; i < SEG; i += q) atomicAdd(&logs[i], l);
        }
        __syncthreads();

        // Anything whose tally reached the threshold is a candidate.
        for (int i = threadIdx.x; i < SEG; i += blockDim.x)
            if (base + i < start + count && logs[i] >= thresh) {
                unsigned idx = atomicAdd(ncand, 1u);
                if (idx < cap) cand[idx] = base + i;
            }
        __syncthreads();
    }
}

static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main() {
    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p < BOUND; p++) {
        bool prime = true;
        for (uint32_t q = 2; q * q <= p; q++) if (p % q == 0) { prime = false; break; }
        if (prime) primes.push_back(p);
    }
    // Every prime power p, p^2, p^3... gets its own walk, each adding log(p) once more.
    uint64_t maxN = START + COUNT - 1;
    std::vector<std::pair<uint64_t, uint8_t>> v;
    for (uint32_t p : primes) {
        uint8_t l = (uint8_t)lround(4 * log2((double)p));   // logs in quarter-bits
        for (uint64_t q = p;; q *= p) { v.push_back({q, l}); if (q > maxN / p) break; }
    }
    std::sort(v.begin(), v.end());
    int npp = (int)v.size(), nsmall = 0;
    while (nsmall < npp && v[nsmall].first < SMALL) nsmall++;
    std::vector<uint64_t> pp(npp); std::vector<uint8_t> lg(npp);
    for (int k = 0; k < npp; k++) { pp[k] = v[k].first; lg[k] = v[k].second; }
    uint32_t thresh = (uint32_t)floor(4 * log2((double)START)) - 16;  // small slack for rounding

    printf("Sieving %llu numbers starting at %llu with %d prime powers\n\n",
           (unsigned long long)COUNT, (unsigned long long)START, npp);

    // ---- CPU sieve: each thread owns a chunk, all your cores ----
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

    // ---- GPU sieve ----
    CHECK(cudaFree(0));
    uint64_t *d_pp, *d_cand; uint8_t* d_lg; unsigned* d_n;
    CHECK(cudaMalloc(&d_pp, npp * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_lg, npp));
    CHECK(cudaMalloc(&d_cand, CAP * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_n, sizeof(unsigned)));
    CHECK(cudaMemcpy(d_pp, pp.data(), npp * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lg, lg.data(), npp, cudaMemcpyHostToDevice));
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int blocks = sms * 8;

    // warm-up run on a tiny range so one-time start-up cost isn't timed
    CHECK(cudaMemset(d_n, 0, sizeof(unsigned)));
    sieve_kernel<<<blocks, BLOCK>>>(START, SEG, d_pp, d_lg, npp, nsmall, thresh, d_cand, d_n, CAP);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_n, 0, sizeof(unsigned)));
    auto t1 = std::chrono::steady_clock::now();
    sieve_kernel<<<blocks, BLOCK>>>(START, COUNT, d_pp, d_lg, npp, nsmall, thresh, d_cand, d_n, CAP);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double sieve_s = since(t1);

    unsigned n; CHECK(cudaMemcpy(&n, d_n, sizeof(n), cudaMemcpyDeviceToHost));
    if (n > CAP) { printf("Too many candidates (%u) - raise CAP\n", n); return 1; }
    std::vector<uint64_t> cand(n);
    CHECK(cudaMemcpy(cand.data(), d_cand, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    unsigned long long gpu = 0;
    #pragma omp parallel for reduction(+:gpu)
    for (long long i = 0; i < (long long)n; i++) if (is_smooth(cand[i], primes)) gpu++;
    double gpu_s = since(t1);

    printf("GPU sieve (%d threads): %llu smooth numbers in %.3f s  (sieving %.3f s + %u candidates checked on CPU)\n\n",
           blocks * BLOCK, gpu, gpu_s, sieve_s, n);
    printf("Answers %s.  GPU vs CPU (both sieving): %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    printf("Compare with smooth.cu (dividing): CPU 2.443 s, GPU 0.217 s\n");
    return 0;
}
