# general-alltoallv

Tunable radix-r Bruck algorithms for **MPI_Alltoall** (uniform message size) and
**MPI_Alltoallv** (variable message size), with CPU and GPU implementations and
a side-by-side benchmark harness against the standard MPICH / OpenMPI
algorithms.

## Algorithms

This repo ships two of our own algorithms plus uniform / non-uniform versions
of the most common reference implementations used inside MPICH and OpenMPI.

| Family | File | Variant |
|---|---|---|
| **Ours** | `src/gAta.cpp` | uniform alltoall — `gata_algorithm(r, b, ...)` |
| **Ours** | `src/gAta_gpu.cu` | uniform alltoall, CUDA — `gata_gpu_algorithm` |
| **Ours** | `src/gAtav.cpp` | non-uniform alltoallv — `tuna2_algorithm(r, b, ...)` |
| **Ours** | `src/gAtav_gpu.cu` | non-uniform alltoallv, CUDA — `tuna2_gpu_algorithm` |
| Reference | `benchmarks/OpenMPI_basic_linear_{ata,atav}.cpp` | OpenMPI basic linear |
| Reference | `benchmarks/OpenMPI_pairwise_{ata,atav}.cpp`     | OpenMPI pairwise |
| Reference | `benchmarks/MPICH_scattered_{ata,atav}.cpp`      | MPICH scattered (with `b` batching) |
| Reference | `benchmarks/spreadout_{ata,atav}.cpp`            | spread-out (rotated post order) |
| Reference | `benchmarks/rbruck_ata.cpp`                      | inverse modified r-Bruck (uniform) |
| Reference | `benchmarks/exclusive_or_atav.cpp`               | XOR-based (P must be power of 2) |

Naming convention:

- `_ata.cpp`  → uniform alltoall (scalar `sendcount`)
- `_atav.cpp` → non-uniform alltoallv (`sendcounts[]` array + `sdispls[]`)
- `gAta` / `gata` → our algorithms; `gAtav` / `gatav` → our alltoallv

## Project layout

```
general-alltoallv/
├── Makefile
├── gata_common.h          # shared MPI/STL includes + check_errors + myPow
├── src/                   # core algorithms (CPU + CUDA)
│   ├── gAta.h
│   ├── gAta.cpp / gAta_gpu.cu
│   └── gAtav.cpp / gAtav_gpu.cu
├── benchmarks/            # reference algorithms (MPICH / OpenMPI / etc.)
└── example/               # benchmark drivers
    ├── gata_example.cpp       # CPU,  uniform alltoall comparison
    ├── gata_gpu_example.cu    # GPU,  uniform alltoall
    ├── gatav_example.cpp      # CPU,  non-uniform alltoallv comparison
    └── gatav_gpu_example.cu   # GPU,  non-uniform alltoallv
```

After `make`, all `.o` files and binaries land in `build/`.

## Build

Requires an MPI install (anything supporting `mpicxx`). The GPU targets require
NVCC and a **CUDA-aware MPI** at runtime.

```bash
make ata           # CPU uniform   (build/gata_example,  links 5 reference _ata)
make atav          # CPU alltoallv (build/gatav_example, links 5 reference _atav)
make ata-gpu       # GPU uniform   (build/gata_gpu_example)
make atav-gpu      # GPU alltoallv (build/gatav_gpu_example)
make all           # all four
make clean         # rm -rf build/
```

Override compilers if needed:
```bash
make ata MPICXX=/path/to/mpicxx
make ata-gpu NVCC=/usr/local/cuda-12/bin/nvcc
```

## Run

```bash
# uniform alltoall comparison
mpirun -n 8 ./build/gata_example  <loop_count> <base_list>

# non-uniform alltoallv comparison
mpirun -n 8 ./build/gatav_example <loop_count> <base_list>
```

`base_list` is a space-separated list of radix `r` values to sweep through.
For each `r`, the example also sweeps `b` (bblock) over **powers of 2 plus
`r-1`**:

| `r` | swept `b` values |
|---|---|
| 2   | `{1}` |
| 4   | `{1, 2, 3}` |
| 8   | `{1, 2, 4, 7}` |
| 64  | `{1, 2, 4, 8, 16, 32, 63}` |
| 1024| `{1, 2, 4, 8, …, 512, 1023}` |
| 8192| `{1, 2, 4, …, 4096, 8191}` |

This keeps the sweep at `O(log P)` instead of `O(P)`.

Examples:
```bash
# r=8, n loops 2..1024, 1 iteration each, prints every algorithm + every (r, b)
mpirun -n 8 ./build/gata_example 1 8

# multiple r values
mpirun -n 16 ./build/gata_example 1 4 8 16

# more averaging — 5 iterations per (algorithm, n, b, r)
mpirun -n 64 ./build/gata_example 5 32 64
```

## Output

Every benchmarked call prints one line of CSV-friendly output:

```
[<tag>] nprocs, n, extra1, extra2, max_time_seconds
```

| tag                | extra1 | extra2 | algorithm |
|--------------------|--------|--------|---|
| `MPI_Alltoall(v)`  | 0      | 0      | system MPI default |
| `OMPI-linear`      | 0      | 0      | OpenMPI basic linear |
| `OMPI-pairwise`    | 0      | 0      | OpenMPI pairwise |
| `MPICH-scattered`  | `b`    | 0      | MPICH scattered (b sweep) |
| `Spreadout`        | 0      | 0      | spread-out |
| `rbruck` (ata only) | `r`   | 0      | inverse modified r-Bruck |
| `XOR` (atav only)   | 0     | 0      | XOR (only printed when P is power of 2) |
| `gata`             | `b`    | `r`    | our uniform alltoall |
| `gatav`            | `b`    | `r`    | our non-uniform alltoallv |

`max_time_seconds` is `MPI_Allreduce(MAX)` across ranks of the local
`MPI_Wtime()` interval — so it reflects the slowest rank.

`n` is the per-peer message count in elements (`long long`, so 8 bytes each).
The driver sweeps `n ∈ {2, 4, 8, …, 1024}`.

Correctness is verified after every call:

- Each rank fills its sendbuf so that the block destined for rank `i` contains
  the value `i + rank*10`.
- After alltoall(v), rank `R`'s slot `p` must contain `R + p*10`.
- Any violation prints `[<tag>] rank R n=N has errors`.

`sendbuf` is reset before every call (some reference algorithms — e.g.
`rbruck_ata` — reuse `sendbuf` as scratch), so each algorithm sees the same
input. **Reset is outside the timer.**

## Algorithm parameters

```c
int gata_algorithm(int r, int b,
                   char *sendbuf, int sendcount, MPI_Datatype sendtype,
                   char *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm);
```

| param | role |
|---|---|
| `r` | radix (Bruck base). Clamped to `[2, P-1]`. `r=2` is binary Bruck, `r=P` degenerates to direct exchange. |
| `b` | bblock — max number of non-blocking sends in flight per round. Useful range `[1, r-1]`. Larger `b` = more parallelism, smaller `b` = less network contention. |
| `sendbuf`/`recvbuf` | `nprocs` consecutive blocks of `sendcount` × `sendtype` elements. |

`tuna2_algorithm` has the same `(r, b)` but with `sendcounts[]` / `sdispls[]` /
`recvcounts[]` / `rdispls[]` arrays instead of scalars.

GPU versions (`gata_gpu_algorithm`, `tuna2_gpu_algorithm`) have the same
signatures but expect `sendbuf` / `recvbuf` to be **CUDA device pointers** and
require a CUDA-aware MPI.

## Rough tuning guide

| message size (per peer) | suggested `r` | suggested `b` |
|---|---|---|
| small (< 1 KB)          | high (8, 16, P) | `r-1` (max parallelism) |
| medium                  | 3–8              | mid (~`r/2`)            |
| large (> 1 MB)          | small (2–4)      | small (1–2, less contention) |

The benchmark harness sweeps the full `(r, b)` grid, so the optimum for a
given system + message size shows up directly in the timing table.

## GPU notes

The two `_gpu` examples need:

1. **CUDA-aware MPI**: device pointers can be passed to `MPI_Isend`/`Irecv`.
2. **A reachable GPU per rank**: each rank binds to `cudaSetDevice(rank %
   dev_count)` automatically.

If MPI isn't CUDA-aware you'll typically see a segfault inside MPI when it
dereferences the device pointer. A staging-buffer fallback isn't shipped — let
me know if you want one.
