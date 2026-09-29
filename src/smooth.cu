// smooth.cu - the core job of a sieve, done on CPU and on GPU, side by side.
// Question for each number: "does it break down entirely into small primes?"
// (a "smooth" number - the raw material the Number Field Sieve collects).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

const uint64_t START = 1000000000000ULL;  // start at 10^12
const uint64_t COUNT = 50000000ULL;       // test 50 million numbers
const uint32_t BOUND = 1000;              // "small primes" = primes below 1000

__constant__ uint32_t d_primes[200];      // the small primes, in fast GPU constant memory

// Divide out every small prime. If nothing is left over (n == 1), the number was smooth.
__host__ __device__ inline bool is_smooth(uint64_t n, const uint32_t* primes, int np) {
    for (int i = 0; i < np; i++) {
        uint32_t p = primes[i];
        while (n % p == 0) n /= p;
        if (n == 1) return true;
    }
    return n == 1;
}

// GPU version: thousands of threads, each takes a slice of the numbers.
__global__ void count_smooth(uint64_t start, uint64_t count, int np, unsigned long long* result) {
    unsigned long long local = 0;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        if (is_smooth(start + i, d_primes, np)) local++;
    atomicAdd(result, local);
}

static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main() {
    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p < BOUND; p++) {
        bool prime = true;
        for (uint32_t q = 2; q * q <= p; q++) if (p % q == 0) { prime = false; break; }
        if (prime) primes.push_back(p);
    }
    int np = (int)primes.size();
    printf("Testing %llu numbers starting at %llu\nfor smoothness over the %d primes below %u\n\n",
           (unsigned long long)COUNT, (unsigned long long)START, np, BOUND);

    // ---- CPU: all your cores ----
    auto t0 = std::chrono::steady_clock::now();
    unsigned long long cpu = 0;
    #pragma omp parallel for reduction(+:cpu) schedule(static)
    for (long long i = 0; i < (long long)COUNT; i++)
        if (is_smooth(START + i, primes.data(), np)) cpu++;
    double cpu_s = seconds_since(t0);
    printf("CPU (%2d threads): %llu smooth numbers in %.3f s\n", omp_get_max_threads(), cpu, cpu_s);

    // ---- GPU ----
    CHECK(cudaFree(0));  // wake the GPU up first so start-up time isn't counted
    CHECK(cudaMemcpyToSymbol(d_primes, primes.data(), np * sizeof(uint32_t)));
    unsigned long long* d_result;
    CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_result, 0, sizeof(unsigned long long)));
    int sms;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    auto t1 = std::chrono::steady_clock::now();
    count_smooth<<<sms * 32, 256>>>(START, COUNT, np, d_result);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double gpu_s = seconds_since(t1);

    unsigned long long gpu;
    CHECK(cudaMemcpy(&gpu, d_result, sizeof(gpu), cudaMemcpyDeviceToHost));
    printf("GPU (%d threads): %llu smooth numbers in %.3f s\n\n", sms * 32 * 256, gpu, gpu_s);

    printf("Answers %s.  GPU speedup: %.1fx\n", cpu == gpu ? "MATCH" : "DO NOT MATCH", cpu_s / gpu_s);
    cudaFree(d_result);
    return 0;
}
