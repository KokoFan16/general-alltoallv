/*
 * MPICH_scattered_ata.cpp
 *
 * Uniform alltoall version of MPICH's scattered algorithm.
 * Posts at most `b` Isends/Irecvs at a time, with destinations rotated
 * by rank to spread the load.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

int MPICH_intra_scattered_ata(int b,
                              char *sendbuf, int sendcount, MPI_Datatype sendtype,
                              char *recvbuf, int recvcount, MPI_Datatype recvtype,
                              MPI_Comm comm)
{
    int rank, comm_size, send_extent, recv_extent;
    int mpi_errno = MPI_SUCCESS;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Type_size(sendtype, &send_extent);
    MPI_Type_size(recvtype, &recv_extent);

    if (b <= 0 || b > comm_size) b = comm_size;

    MPI_Request *reqarray = (MPI_Request *) malloc(2 * b * sizeof(MPI_Request));
    MPI_Status  *starray  = (MPI_Status  *) malloc(2 * b * sizeof(MPI_Status));

    // Post b Isends/Irecvs at a time
    for (int ii = 0; ii < comm_size; ii += b) {
        int req_cnt = 0;
        int ss = comm_size - ii < b ? comm_size - ii : b;

        for (int i = 0; i < ss; i++) {
            int dst = (rank + i + ii) % comm_size;
            mpi_errno = MPI_Irecv(&recvbuf[(size_t)dst * recvcount * recv_extent],
                                  recvcount, recvtype, dst, 0, comm,
                                  &reqarray[req_cnt++]);
            if (mpi_errno != MPI_SUCCESS) { free(reqarray); free(starray); return -1; }
        }

        for (int i = 0; i < ss; i++) {
            int dst = (rank - i - ii + comm_size) % comm_size;
            mpi_errno = MPI_Isend(&sendbuf[(size_t)dst * sendcount * send_extent],
                                  sendcount, sendtype, dst, 0, comm,
                                  &reqarray[req_cnt++]);
            if (mpi_errno != MPI_SUCCESS) { free(reqarray); free(starray); return -1; }
        }

        mpi_errno = MPI_Waitall(req_cnt, reqarray, starray);
        if (mpi_errno != MPI_SUCCESS) { free(reqarray); free(starray); return -1; }
    }

    free(reqarray);
    free(starray);
    return 0;
}
