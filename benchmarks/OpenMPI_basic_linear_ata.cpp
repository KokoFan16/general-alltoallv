/*
 * OpenMPI_basic_linear_ata.cpp
 *
 * Uniform alltoall version of OpenMPI's basic linear algorithm.
 * Each rank posts P-1 Irecvs from every peer, then P-1 Isends, then waits.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

int ompi_alltoall_intra_basic_linear(char *sendbuf, int sendcount, MPI_Datatype sendtype,
                                     char *recvbuf, int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm)
{
    int rank, size, sext, rext, err, nreqs;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &sext);
    MPI_Type_size(recvtype, &rext);

    // Handle send to self first
    memcpy(&recvbuf[(size_t)rank * recvcount * rext],
           &sendbuf[(size_t)rank * sendcount * sext],
           (size_t)recvcount * rext);

    if (size == 1) return MPI_SUCCESS;

    MPI_Request *preq = (MPI_Request *) malloc(2 * (size - 1) * sizeof(MPI_Request));
    nreqs = 0;

    // Post all receives first
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        err = MPI_Irecv(&recvbuf[(size_t)i * recvcount * rext],
                        recvcount, recvtype, i, 0, comm, &preq[nreqs++]);
        if (err != MPI_SUCCESS) { free(preq); return -1; }
    }

    // Then post all sends
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        err = MPI_Isend(&sendbuf[(size_t)i * sendcount * sext],
                        sendcount, sendtype, i, 0, comm, &preq[nreqs++]);
        if (err != MPI_SUCCESS) { free(preq); return -1; }
    }

    err = MPI_Waitall(nreqs, preq, MPI_STATUSES_IGNORE);
    free(preq);

    return err;
}
