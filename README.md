# RSA Factoring Lab

![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&logoColor=white) ![C++](https://img.shields.io/badge/C++-OpenMP-00599C?logo=cplusplus&logoColor=white)

A lab for learning how RSA numbers get factored. I ran CADO-NFS, the tool used for the factoring records, on a home PC, then wrote a GPU sieve from scratch and tuned it with a profiler. I read about RSA-260 being factored and wanted to see how far a home PC could get.

**Hardware:** Intel i5-12600K · NVIDIA RTX 3060 Ti (8 GB, sm_86) · Windows + WSL2 Ubuntu · CUDA 13.3

## CADO-NFS runs

| Target | Digits | Wall time | CPU time |
|---|---|---|---|
| Warm-up semiprime | 60 | 13 s | 45 CPU-s |
| RSA-100 | 100 | 286 s | 2,782 CPU-s |

## GPU sieve versions

Benchmark: count numbers in `[10^12, 10^12 + COUNT)` whose prime factors are all below 1000.

| COUNT | Correct answer |
|---|---|
| 50,000,000 | 211,156 |
| 500,000,000 | 2,112,610 |
| 2,000,000,000 | 8,448,866 |

| File | Change | Result |
|---|---|---|
| `smooth.cu` | Brute-force division | GPU 0.217 s vs CPU 2.443 s at 50M |
| `sieve.cu` | Sieve instead of divide | CPU 136× faster than dividing; GPU 1.1× |
| `sieve2.cu` | GPU checks its own candidates | GPU 0.612 s vs CPU 0.543 s at 2B |
| `sieve3.cu` | Division-free check + candidate compaction | GPU 0.149 s vs CPU 0.347 s at 2B |
| `sieve4.cu` | Trailing-zero count for powers of 2 | GPU 0.144 s, CPU 0.419 s at 2B |
| `sieve5.cu` | Profiler-guided tuning | GPU 0.132 s vs CPU 0.338 s at 2B |

## What I learned

Every version answers the same question and has to match the counts above. Each file's header comment explains its idea in more detail.

**`smooth.cu`: divide everything.** Each number is divided by every prime below 1000. It's the obvious way, and it's slow. At 50M numbers the GPU took 0.217 s and the CPU 2.443 s.

**`sieve.cu`: sieve instead of divide.** The numbers divisible by 7 are every 7th number. So each prime walks down the range in steps of p and adds log(p) to every number it lands on. A number whose tallies add up to its own size is built only from small primes. No division needed for the heavy lifting.

This was the surprise. On the CPU, sieving was 136× faster than dividing. On the GPU it was only 1.1× faster. The GPU sieved in 3 ms, then sat idle while 211,156 candidates were copied back and checked on the CPU, so the whole run took 15 ms. The algorithm got better, but the bottleneck just moved.

**`sieve2.cu`: the GPU checks its own candidates.** Each GPU block confirms its own candidates, so only one number crosses back to the CPU. At 2B numbers the GPU lost: 0.612 s against the CPU's 0.543 s. The check used 64-bit division, which is the GPU's weakest skill, and one candidate made its whole 32-thread warp wait.

**`sieve3.cu`: no division, and packed work.** For an odd prime p, multiplying by p's inverse mod 2^64 and doing one compare tells you whether p divides n. Each block also gathers its candidates into a short list first, so busy threads aren't scattered across warps. The CPU got the division-free check too, to keep the race fair. At 2B: GPU 0.149 s, CPU 0.347 s.

**`sieve4.cu`: two fixes that barely mattered.** Powers of 2 are counted from the number's trailing zero bits instead of walked, which avoids shared-memory bank conflicts. Walkers also carry their position from one segment to the next instead of recomputing it with a division. The GPU went from 0.149 s to 0.144 s, about 3%, which is noise. The CPU got slower, 0.419 s: it split the work into equal fixed slices, and the slower E-cores held everyone up.

**`sieve5.cu`: let the profiler pick.** Nsight Compute on `sieve4` pointed at three things:

- Occupancy was 33%. Each block needed 44 KB of shared memory, so only 2 blocks fit per SM.
- 40% of stall time was warps waiting at `__syncthreads()` for slower warps.
- Only about 18 of 32 threads per warp were active on average.

The fixes: every tally fits in one byte, so 4 are packed into each 32-bit word, which brings a block down to about 16 KB. The launch is exactly one wave of blocks. And each candidate is checked by a whole warp, with the 32 threads splitting the 167 odd primes between them. The CPU went back to sharing work on demand. At 2B: GPU 0.132 s, CPU 0.338 s.

| Metric (Nsight Compute) | `sieve4` | `sieve5` |
|---|---|---|
| Achieved occupancy | 32% (limited by shared memory) | 64% (limited by registers) |
| Compute (SM) throughput | 25% | 66% |
| Top stall | `__syncthreads()`, 40% of stalls | Branch resolution, 31% |
| Active threads per warp | 17.9 / 32 | 20.3 / 32 |
| Cycles for 50M numbers | 6.34M | 5.79M |

The GPU went from mostly waiting to mostly working. At 50M numbers it went from 0.217 s in `smooth.cu` to 0.004 s, about 54×.

What I'd take away:

- The algorithm mattered more than the hardware. Sieving gave 136×. The GPU gave about 11×.
- Every fix moved the bottleneck somewhere else: CPU checking, then GPU division, then idle threads, then occupancy.
- The two fixes in `sieve4` that didn't come from the profiler barely mattered. The profiler found the real limits on its first run.
- On a CPU with P-cores and E-cores, equal fixed slices let the slow cores set the pace.
- GPU clocks vary between runs, so compare cycles, not milliseconds, across profiled runs.

## Setup (Windows + WSL2)

```powershell
wsl --install -d Ubuntu
wsl --set-default Ubuntu
```

Inside Ubuntu:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit build-essential
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

## Build and run

```bash
make                    # edit ARCH in the Makefile for non-Ampere GPUs
./sieve5 2000000000     # optional COUNT (sieve2 onward)
```

## Profile

Enable GPU performance counters in NVIDIA Control Panel (Desktop → Enable Developer Settings, then Developer → Manage GPU Performance Counters → Allow access to all users), run `wsl --shutdown`, then:

```bash
ncu --launch-skip 1 --launch-count 1 --section SpeedOfLight --section Occupancy --section WarpStateStats ./sieve5 50000000
```

## CADO-NFS

```bash
sudo apt install -y cmake git python3 python3-flask python3-requests libgmp-dev libhwloc-dev
git config --global http.version HTTP/1.1
git clone --depth 1 https://gitlab.inria.fr/cado-nfs/cado-nfs.git && cd cado-nfs && make -j$(nproc)
./cado-nfs.py 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139   # RSA-100
```

## Roadmap: quadratic sieve

- [ ] CPU quadratic sieve that cracks a 30-digit number
- [ ] GPU sieve with the small prime variation
- [ ] Multiple polynomials (SIQS) to reach 50–60 digits
- [ ] Race it against CADO-NFS
- [ ] Profile and tune

## Sources

- [Factoring RSA-260, Cognition](https://cognition.com/blog/factoring-rsa-260)
- [RSA-896, Stephen Weis](https://saweis.net/posts/rsa-896.html)
- [RSA numbers, Wikipedia](https://en.wikipedia.org/wiki/RSA_numbers)
- [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs)
- [Profiling CUDA programs on WSL 2, Peter Chng](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/)

## How it was built

I built this with Claude as a pair programmer. I ran the CADO-NFS and GPU experiments on my own hardware, read the profiler output to decide what to change next, wrote parts of the code, and reviewed the rest.
