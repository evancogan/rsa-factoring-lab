# RSA Factoring Lab

> **Work in progress.** Full write-up coming, check back later.

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
