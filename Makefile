NVCC  ?= nvcc
ARCH  ?= sm_86
FLAGS  = -O3 -arch=$(ARCH) -Xcompiler -fopenmp -lgomp
PROGS  = smooth sieve sieve2 sieve3 sieve4 sieve5

all: $(PROGS)

%: src/%.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(PROGS)

.PHONY: all clean
