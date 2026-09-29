# RSA Factoring Lab

A hands-on project to understand how RSA numbers get broken. It started with running the real record-setting tool (CADO-NFS) on a home PC. Then it moved on to building a GPU sieve from scratch and tuning it with a profiler.

**Hardware:** Intel i5-12600K (6 P-cores + 4 E-cores, 16 threads) · NVIDIA RTX 3060 Ti (8 GB, 38 SMs, sm_86) · Windows + WSL2 Ubuntu · CUDA 13.3

---

## The big picture

RSA security rests on one bet: multiplying two huge primes is easy, but splitting the product back apart is not. The best known classical method is the **Number Field Sieve (NFS)**. It has five stages:

1. **Polynomial selection.** Choose a formula that makes useful numbers common.
2. **Sieving.** Collect *relations*, numbers that break down entirely into small primes ("smooth" numbers). This is where most of the time goes.
3. **Filtering.** Remove duplicates and relations that can never pair up.
4. **Linear algebra.** Pick a subset where every prime appears an even number of times, so the product is a perfect square.
5. **Square root.** Two squares with the same remainder mod N give up a factor via a gcd.

The key moment is when the number of relations passes the number of primes involved (CADO-NFS reports this as **"excess"**). From then on, a perfect-square combination is guaranteed to exist, by the pigeonhole principle.

## Where the world is (September 2026)

| Number | Size | Status |
|---|---|---|
| RSA-250 | 829 bits | Factored 2020, about 2,700 CPU core-years |
| **RSA-260** | 862 bits | Factored **Sept 3, 2026** by Eric Lu (Cognition): GPU port of CADO-NFS written largely by Devin agents, about 4,900 GPU-days, about $400k at market rates |
| **RSA-896** | 896 bits | Factored **Sept 19, 2026** by Stephen Weis: Claude-assisted GPU port of CADO-NFS, about 2,048 GPUs for 10 days |
| **RSA-270** | 895 bits | **Smallest unfactored RSA challenge number** |
| RSA-2048 | 2048 bits | About a billion times harder than RSA-260; out of classical reach |

Neither GPU port has been released publicly. The RSA-260 writeup reports sieving at about 77% of total compute, and its central engineering rule was **"make GPU never block on CPU."**

**Quantum side note:** Shor's algorithm turns factoring into *period finding*, which quantum computers do efficiently. The largest number honestly factored with Shor on real hardware is **21**. Google's 2025 estimate for RSA-2048 is about 1 million noisy qubits running for about a week.

---

## Part 1: Running CADO-NFS at home

Built CADO-NFS from source in WSL Ubuntu and factored:

| Target | Digits | Wall time | CPU time | Notes |
|---|---|---|---|---|
| Warm-up semiprime | 60 | **13 s** | 45 CPU-s | 49,410 relations; "excess 160"; 6,865-row matrix |
| **RSA-100** | 100 | **286 s** | 2,782 CPU-s | First factored in 1991; about 22× harder than c60 |

For scale, RSA-250 took about **30 million times** the compute of this RSA-100 run.

---

## Part 2: Building a GPU sieve from scratch

**Benchmark task:** count the numbers in `[10^12, 10^12 + COUNT)` that are built entirely from primes below 1000. Every version must produce the same answers:

| COUNT | Correct answer |
|---|---|
| 50,000,000 | 211,156 |
| 500,000,000 | 2,112,610 |
| 2,000,000,000 | 8,448,866 |

### The versions

| File | Idea | Result |
|---|---|---|
| `smooth.cu` | Brute force: divide every number by every small prime | GPU 0.217 s vs CPU 2.443 s at 50M: **11.3×** |
| `sieve.cu` | **Sieve instead of divide:** each prime "walks" in steps of p and adds log(p) to a tally | CPU **136× faster** than dividing. GPU sieving took only 3 ms, but the total was 15 ms because candidates were checked on the CPU (**1.1×**) |
| `sieve2.cu` | GPU checks its own candidates, so only one number crosses back | GPU *lost* at scale (0.612 s vs CPU 0.543 s at 2B). The check used 64-bit division, and one candidate stalled its whole 32-thread warp |
| `sieve3.cu` | **Division-free check** (multiply by the modular inverse of p) + **candidate compaction** | 2B: GPU **0.149 s** vs CPU 0.347 s, **2.3×** |
| `sieve4.cu` | Powers of 2 via trailing-zero count; walkers carry their position between segments | GPU 0.144 s (about 3%, noise). CPU got *slower* (0.419 s): fixed slicing let the E-cores hold everyone up |
| `sieve5.cu` | **Profiler-guided:** byte-packed tallies, exactly one wave of blocks, whole warps check each candidate | 2B: GPU **0.132 s** vs CPU 0.338 s, **2.6×** |

Overall, the same question on the same GPU went from 0.217 s to 0.004 s at 50M, about **54×**.

### What the profiler (Nsight Compute) showed

| Metric | sieve4 | sieve5 |
|---|---|---|
| Occupancy (achieved) | 32% (limited by 44 KB shared memory/block) | **64%** (now limited by registers) |
| Compute (SM) throughput | 25% | **66%** |
| Top stall | waiting at `__syncthreads()` (40% of stalls) | branch resolution (31%) |
| Active threads per warp | 17.9 / 32 | 20.3 / 32 |
| Cycles for 50M numbers | 6.34M | **5.79M** |

The GPU went from **mostly waiting** to **mostly working**. From here, bigger gains require doing *less work per number*, not waiting less.

### Lessons

1. **Algorithm beats hardware.** Switching from dividing to sieving gave 136×. The GPU alone gave 11×.
2. **The bottleneck always moves.** Fix one step and another becomes the limit: CPU checking, then GPU division, then warp divergence, then occupancy.
3. **GPUs hate division and branching.** Replace divides with multiplies and pack irregular work together.
4. **Measure, don't guess.** Two of my hand-picked optimizations (bank conflicts, recomputed offsets) barely mattered. The profiler found the real limiters (occupancy, barrier stalls) on its first run.
5. **Hybrid CPUs need dynamic scheduling.** Equal fixed slices let the slow E-cores set the pace.
6. **Compare cycles, not milliseconds, across profiled runs.** GPU clocks vary between runs.

---

## Reproducing

### Setup (Windows + WSL2)

```powershell
wsl --install -d Ubuntu
wsl --set-default Ubuntu     # needed if Docker Desktop's distro became the default
```

Inside Ubuntu, install the CUDA toolkit from NVIDIA's **WSL** repo. This package has no driver, because WSL uses the Windows driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit build-essential
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

### Build and run

```bash
make                    # builds all versions (edit ARCH in the Makefile for non-Ampere GPUs)
./sieve5 2000000000     # any version takes an optional COUNT (sieve2 onward)
```

### Profile

First, on Windows, allow profiler access in NVIDIA Control Panel: **Desktop → Enable Developer Settings**, then **Developer → Manage GPU Performance Counters → Allow access to all users**. Run `wsl --shutdown` afterwards so WSL picks up the change. Then:

```bash
ncu --launch-skip 1 --launch-count 1 --section SpeedOfLight --section Occupancy --section WarpStateStats ./sieve5 50000000
```

### CADO-NFS

```bash
sudo apt install -y cmake git python3 python3-flask python3-requests libgmp-dev libhwloc-dev
git config --global http.version HTTP/1.1
git clone --depth 1 https://gitlab.inria.fr/cado-nfs/cado-nfs.git && cd cado-nfs && make -j$(nproc)
./cado-nfs.py 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139   # RSA-100
```

---

## Roadmap: Goal 2, our own quadratic sieve

The quadratic sieve heads for the same goal as NFS (two matching squares) but with simpler math. It's the fastest method up to about 100 digits. It sieves Q(x) = x² − N for x just above √N. These values are small, so smooth ones are common, and each prime walks from **two** starting points.

- [ ] **2.1** CPU quadratic sieve that cracks a 30-digit number end to end
- [ ] **2.2** Swap in the GPU sieve, with the *small prime variation* (skip tiny primes, lower the threshold)
- [ ] **2.3** Many polynomials (SIQS) to reach 50–60 digits
- [ ] **2.4** Race it against CADO-NFS on the same numbers
- [ ] **2.5** Profile and tune, with Claude Code in the role Devin played

Side quest: simulate Shor's algorithm on the GPU. An 8 GB card holds about 28–29 simulated qubits.

---

## Sources

- [Factoring RSA-260, Cognition](https://cognition.com/blog/factoring-rsa-260)
- [RSA-896, Stephen Weis](https://saweis.net/posts/rsa-896.html)
- [RSA numbers, Wikipedia](https://en.wikipedia.org/wiki/RSA_numbers)
- [Integer factorization records, Wikipedia](https://en.wikipedia.org/wiki/Integer_factorization_records)
- [Tracking the Cost of Quantum Factoring, Google](https://blog.google/security/tracking-cost-of-quantum-factori/)
- [The CRQC Reality Check, Aethyr Research](https://aethyrresearch.com/blog/crqc-reality-check-2026)
- [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs)
- [Factoring Integers With Shor's Algorithm, NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/applications/python/shors.html)
- [Profiling CUDA programs on WSL 2, Peter Chng](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/)
