/*
 * gAtav_gpu.cu
 *
 * GPU version of tuna2_algorithm.
 *
 * Assumptions:
 *   - sendbuf and recvbuf are CUDA device pointers.
 *   - The MPI implementation is CUDA-aware (device pointers can be passed
 *     directly to MPI_Isend / MPI_Irecv / MPI_Sendrecv).
 *   - sendcounts, sdispls, recvcounts, rdispls remain on the host.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "gAta.h"

#define CUDA_CHECK(stmt) do {                                          \
    cudaError_t err = (stmt);                                          \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n';   \
        return 1;                                                      \
    }                                                                  \
} while (0)

int tuna2_gpu_algorithm(int r, int b,
                        char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                        char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                        MPI_Comm comm) {

    if (r < 2) r = 2;

    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(sendtype, &typesize);

    if (r > nprocs - 1) r = nprocs - 1;
    if (b <= 0 || b > nprocs) b = nprocs;

    int w, max_rank, nlpow, d, K, i, num_reqs;
    int local_max_count = 0, max_send_count = 0;
    int rotate_index_array[nprocs];
    w = 0; nlpow = 1; max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }
    for (i = 0; i < w - 1; i++) nlpow *= r;
    d = (nlpow * r - nprocs) / nlpow;
    K = w * (r - 1) - d;

    int rem1 = K + 1, rem2 = r + 1;
    int sendNcopy[nprocs - rem1];
    char *extra_buffer = nullptr, *temp_recv_buffer = nullptr, *temp_send_buffer = nullptr;
    int extra_ids[nprocs - rem2];
    memset(extra_ids, -1, sizeof(extra_ids));
    int spoint = 1, distance = 1, next_distance = distance * r, di = 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (K < nprocs - 1) {
        for (i = 0; i < nprocs; i++) {
            if (sendcounts[i] > local_max_count) local_max_count = sendcounts[i];
        }
        MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

        for (i = 0; i < nprocs; i++) {
            rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
        }

        CUDA_CHECK(cudaMalloc((void**)&extra_buffer,
                              (size_t)max_send_count * typesize * (nprocs - rem1)));
        CUDA_CHECK(cudaMalloc((void**)&temp_recv_buffer,
                              (size_t)max_send_count * nprocs * typesize));
        CUDA_CHECK(cudaMalloc((void**)&temp_send_buffer,
                              (size_t)max_send_count * nprocs * typesize));

        for (int x = 0; x < w; x++) {
            for (int z = 1; z < r; z++) {
                spoint = z * distance;
                if (spoint > nprocs) break;
                int end = (spoint + distance > nprocs) ? nprocs : spoint + distance;
                for (i = spoint + 1; i < end; i++) {
                    extra_ids[i - rem2] = di++;
                }
            }
            distance *= r;
        }
    }

    // copy data destined for self (device-to-device)
    CUDA_CHECK(cudaMemcpyAsync(&recvbuf[rdispls[rank] * typesize],
                               &sendbuf[sdispls[rank] * typesize],
                               (size_t)recvcounts[rank] * typesize,
                               cudaMemcpyDeviceToDevice, stream));

    int sent_blocks[r - 1][nlpow];
    int metadata_recv[r - 1][nlpow];
    int nc, rem, ns, ze, ss;
    spoint = 1; distance = 1; next_distance = distance * r;

    MPI_Request *reqs = (MPI_Request*) malloc(2 * r * sizeof(MPI_Request));
    MPI_Status  *stats = (MPI_Status*)  malloc(2 * r * sizeof(MPI_Status));
    if (!reqs || !stats) {
        std::cerr << "MPI_Request/Status allocation failed!\n";
        return 1;
    }

    int metadata_send[nlpow];
    int comm_size[r - 1];

    for (int x = 0; x < w; x++) {
        ze = (x == w - 1) ? r - d : r;
        int zoffset = 0, zc = ze - 1;
        int zns[zc];

        for (int k = 1; k < ze; k += b) {
            ss = ze - k < b ? ze - k : b;
            num_reqs = 0;
            int send_zoffset = 0;

            for (int s = 0; s < ss; s++) {
                int z = k + s;

                spoint = z * distance;
                nc = nprocs / next_distance * distance;
                rem = nprocs % next_distance - spoint;
                if (rem < 0) rem = 0;
                ns = (rem > distance) ? (nc + distance) : (nc + rem);
                zns[z - 1] = ns;

                int recvrank = (rank + spoint) % nprocs;
                int sendrank = (rank - spoint + nprocs) % nprocs;

                if (ns == 1) {
                    MPI_Irecv(&recvbuf[rdispls[recvrank] * typesize],
                              recvcounts[recvrank] * typesize, MPI_CHAR,
                              recvrank, 1, comm, &reqs[num_reqs++]);
                    MPI_Isend(&sendbuf[sdispls[sendrank] * typesize],
                              sendcounts[sendrank] * typesize, MPI_CHAR,
                              sendrank, 1, comm, &reqs[num_reqs++]);
                } else {
                    di = 0;
                    for (int ii = spoint; ii < nprocs; ii += next_distance) {
                        int j_end = (ii + distance > nprocs) ? nprocs : ii + distance;
                        for (int j = ii; j < j_end; j++) {
                            int id = (j + rank) % nprocs;
                            sent_blocks[z - 1][di++] = id;
                        }
                    }

                    int sendCount = 0, offset = 0;
                    for (int ii = 0; ii < di; ii++) {
                        int send_index = rotate_index_array[sent_blocks[z - 1][ii]];
                        int o = (sent_blocks[z - 1][ii] - rank + nprocs) % nprocs - rem2;

                        if (ii % distance == 0)
                            metadata_send[ii] = sendcounts[send_index];
                        else
                            metadata_send[ii] = sendNcopy[extra_ids[o]];
                        offset += metadata_send[ii] * typesize;
                    }

                    MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0,
                                 metadata_recv[z - 1], di, MPI_INT, recvrank, 0,
                                 comm, MPI_STATUS_IGNORE);

                    for (int ii = 0; ii < di; ii++) sendCount += metadata_recv[z - 1][ii];
                    comm_size[z - 1] = sendCount;

                    // prepare send data on device
                    offset = 0;
                    for (int ii = 0; ii < di; ii++) {
                        int send_index = rotate_index_array[sent_blocks[z - 1][ii]];
                        int o = (sent_blocks[z - 1][ii] - rank + nprocs) % nprocs - rem2;
                        size_t size = 0;

                        if (ii % distance == 0) {
                            size = (size_t)sendcounts[send_index] * typesize;
                            CUDA_CHECK(cudaMemcpyAsync(
                                &temp_send_buffer[send_zoffset + offset],
                                &sendbuf[sdispls[send_index] * typesize],
                                size, cudaMemcpyDeviceToDevice, stream));
                        } else {
                            size = (size_t)sendNcopy[extra_ids[o]] * typesize;
                            CUDA_CHECK(cudaMemcpyAsync(
                                &temp_send_buffer[send_zoffset + offset],
                                &extra_buffer[(size_t)extra_ids[o] * max_send_count * typesize],
                                size, cudaMemcpyDeviceToDevice, stream));
                        }
                        offset += size;
                    }

                    // make sure all device copies for this z are done before MPI sees the buffer
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    MPI_Irecv(&temp_recv_buffer[zoffset], comm_size[z - 1] * typesize,
                              MPI_CHAR, recvrank, recvrank + z, comm, &reqs[num_reqs++]);
                    MPI_Isend(&temp_send_buffer[send_zoffset], offset, MPI_CHAR,
                              sendrank, rank + z, comm, &reqs[num_reqs++]);

                    zoffset += comm_size[z - 1] * typesize;
                    send_zoffset += offset;
                }
            }

            MPI_Waitall(num_reqs, reqs, stats);
            for (int ii = 0; ii < num_reqs; ii++) {
                if (stats[ii].MPI_ERROR != MPI_SUCCESS) {
                    std::cerr << "Request " << ii << " error: "
                              << stats[ii].MPI_ERROR << '\n';
                }
            }
            MPI_Barrier(comm);
        }

        if (K < nprocs - 1) {
            int offset = 0;
            for (int ii = 0; ii < zc; ii++) {
                for (int j = 0; j < zns[ii]; j++) {
                    if (zns[ii] > 1) {
                        size_t size = (size_t)metadata_recv[ii][j] * typesize;
                        int o = (sent_blocks[ii][j] - rank + nprocs) % nprocs - rem2;

                        if (j < distance) {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &recvbuf[rdispls[sent_blocks[ii][j]] * typesize],
                                &temp_recv_buffer[offset],
                                size, cudaMemcpyDeviceToDevice, stream));
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &extra_buffer[(size_t)extra_ids[o] * max_send_count * typesize],
                                &temp_recv_buffer[offset],
                                size, cudaMemcpyDeviceToDevice, stream));
                            sendNcopy[extra_ids[o]] = metadata_recv[ii][j];
                        }
                        offset += size;
                    }
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        distance *= r;
        next_distance *= r;
    }

    if (K < nprocs - 1) {
        cudaFree(extra_buffer);
        cudaFree(temp_recv_buffer);
        cudaFree(temp_send_buffer);
    }
    free(reqs);
    free(stats);
    cudaStreamDestroy(stream);

    return 0;
}
