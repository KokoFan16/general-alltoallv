/*
 * scattered.cpp
 *
 *  Created on: Feb 19, 2024
 *      Author: kokofan
 */


#include "../gata_common.h"

int MPICH_intra_scattered(int b, char *sendbuf, int *sendcounts, int *sdispls,
						  MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
						  MPI_Datatype recvtype, MPI_Comm comm)
{

    int comm_size, i;
    int send_extent, recv_extent;
    int dst, rank, req_cnt;
    MPI_Request *reqarray;
    MPI_Status *starray;
    int ii, ss;
    int mpi_errno = MPI_SUCCESS;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    /* Get extent of send and recv types */
    MPI_Type_size(sendtype, &send_extent);
    MPI_Type_size(recvtype, &recv_extent);


    if (b <= 0 || b > comm_size) b = comm_size;

    reqarray = (MPI_Request *)malloc(2 * b * sizeof(MPI_Request));
    starray = (MPI_Status *)malloc(2 * b * sizeof(MPI_Status));

    /* post only b isends/irecvs at a time as suggested by Tony Ladd */
	for (ii = 0; ii < comm_size; ii += b) {
		req_cnt = 0;
		ss = comm_size - ii < b ? comm_size - ii : b;

		/* do the communication -- post ss sends and receives: */
		for (i = 0; i < ss; i++) {
			dst = (rank + i + ii) % comm_size;

			if (recvcounts[dst]) {
				mpi_errno = MPI_Irecv((char *) recvbuf + rdispls[dst] * recv_extent,
									   recvcounts[dst], recvtype, dst,
									   0, comm, &reqarray[req_cnt]);

				if (mpi_errno != MPI_SUCCESS) {return -1;}
				req_cnt++;
			}
		}

		for (i = 0; i < ss; i++) {
			dst = (rank - i - ii + comm_size) % comm_size;
			if (sendcounts[dst]) {
				mpi_errno = MPI_Isend((char *) sendbuf + sdispls[dst] * send_extent,
									   sendcounts[dst], sendtype, dst,
									   0, comm, &reqarray[req_cnt]);

				if (mpi_errno != MPI_SUCCESS) {return -1;}
				req_cnt++;
			}
		}

		mpi_errno = MPI_Waitall(req_cnt, reqarray, starray);
		if (mpi_errno != MPI_SUCCESS) {return -1;}
	}

	free(reqarray);
	free(starray);

	return 0;
}
