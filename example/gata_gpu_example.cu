/*
 * gata_gpu_example.cu
 *
 * Example driver for tuna2_gpu_algorithm. Allocates host send/recv buffers
 * for initialization and verification, mirrors them onto the device, runs
 * the GPU alltoallv, then copies the result back to verify.
 *
 * Assumes a CUDA-aware MPI is available.
 */

#include <cuda_runtime.h>
#include "../src/gAta.h"

#define CUDA_CHECK(stmt) do {                                          \
    cudaError_t _e = (stmt);                                           \
    if (_e != cudaSuccess) {                                           \
        std::cerr << "CUDA error " << cudaGetErrorString(_e)           \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n';   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                  \
} while (0)

static int rank, nprocs;
static void run_gata_gpu(int loopcount, int nprocs, std::vector<int> bases);

int main(int argc, char **argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n";
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0)
            std::cout << "Usage: mpirun -n <nprocs> " << argv[0]
                      << " <loop-count> <base-list>\n";
        MPI_Finalize();
        return -1;
    }

    int loopCount = atoi(argv[1]);
    std::vector<int> bases;
    for (int i = 2; i < argc; i++) bases.push_back(atoi(argv[i]));

    // Bind each rank to a GPU (round-robin across visible devices on the node).
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count > 0) {
        CUDA_CHECK(cudaSetDevice(rank % dev_count));
    }

    run_gata_gpu(loopCount, nprocs, bases);

    MPI_Finalize();
    return 0;
}

static void run_gata_gpu(int loopcount, int nprocs, std::vector<int> bases) {

    int mpi_errno = MPI_SUCCESS;
    int basecount = bases.size();

    for (int n = 1024; n <= 1024; n = n * 2) {

        int sendcounts[nprocs];
        memset(sendcounts, 0, nprocs * sizeof(int));
        int sdispls[nprocs];
        int soffset = 0;

        srand(time(NULL));
        for (int i = 0; i < nprocs; i++) sendcounts[i] = n;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));

        for (int i = 0; i < nprocs; i++) {
            sdispls[i] = soffset;
            soffset += sendcounts[i];
        }

        int recvcounts[nprocs];
        MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
        int rdispls[nprocs];
        int roffset = 0;
        for (int i = 0; i < nprocs; i++) {
            rdispls[i] = roffset;
            roffset += recvcounts[i];
        }

        // Host buffers (for init & verification)
        long long *h_send = new long long[soffset];
        long long *h_recv = new long long[roffset];

        int index = 0;
        for (int i = 0; i < nprocs; i++)
            for (int j = 0; j < sendcounts[i]; j++)
                h_send[index++] = i + rank * 10;

        // Device buffers
        long long *d_send = nullptr;
        long long *d_recv = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_send, soffset * sizeof(long long)));
        CUDA_CHECK(cudaMalloc((void**)&d_recv, roffset * sizeof(long long)));
        CUDA_CHECK(cudaMemcpy(d_send, h_send, soffset * sizeof(long long),
                              cudaMemcpyHostToDevice));

        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < basecount; i++) {
            int b = 2;
            for (int it = 0; it < loopcount; it++) {
                double st = MPI_Wtime();
                mpi_errno = tuna2_gpu_algorithm(bases[i], b,
                        (char*)d_send, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
                        (char*)d_recv, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
                        MPI_COMM_WORLD);
                double et = MPI_Wtime();
                double total_time = et - st;

                if (mpi_errno != MPI_SUCCESS)
                    std::cout << "tuna2_gpu_algorithm fail!\n";

                // Copy result back for verification
                CUDA_CHECK(cudaMemcpy(h_recv, d_recv, roffset * sizeof(long long),
                                      cudaMemcpyDeviceToHost));

                int error = check_errors(recvcounts, h_recv, rank, nprocs);
                if (error > 0) {
                    std::cout << rank << " " << n << " " << b
                              << " [gata-GPU] base " << bases[i] << " has errors\n";
                }

                double max_time = 0;
                MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX,
                              MPI_COMM_WORLD);
                if (total_time == max_time) {
                    std::cout << "[gata-GPU] " << nprocs << ", " << n << ", "
                              << b << ", " << bases[i] << ", " << max_time << '\n';
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        cudaFree(d_send);
        cudaFree(d_recv);
        delete[] h_send;
        delete[] h_recv;
    }
}
