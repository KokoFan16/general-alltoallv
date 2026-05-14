/*
 * OpenMPI_pairwise_ata.cpp
 *
 * Uniform alltoall version of OpenMPI's pairwise exchange algorithm.
 * P steps: at step k, rank exchanges with peers (rank+k)%P / (rank-k+P)%P.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

int ompi_alltoall_intra_pairwise(char *sendbuf, int sendcount, MPI_Datatype sendtype,
                                 char *recvbuf, int recvcount, MPI_Datatype recvtype,
                                 MPI_Comm comm)
{
    int rank, size, sext, rext, err = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &sext);
    MPI_Type_size(recvtype, &rext);

    for (int step = 0; step < size; step++) {
        int sendto   = (rank + step) % size;
        int recvfrom = (rank + size - step) % size;

        MPI_Request req = MPI_REQUEST_NULL;

        err = MPI_Irecv(&recvbuf[(size_t)recvfrom * recvcount * rext],
                        recvcount, recvtype, recvfrom, 0, comm, &req);
        if (err != MPI_SUCCESS) return -1;

        err = MPI_Send(&sendbuf[(size_t)sendto * sendcount * sext],
                       sendcount, sendtype, sendto, 0, comm);
        if (err != MPI_SUCCESS) return -1;

        err = MPI_Wait(&req, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) return -1;
    }

    return MPI_SUCCESS;
}
