/*
 * spreadout_atav.cpp
 *
 * Non-uniform spread-out alltoallv. Each rank posts non-blocking recvs from
 * every peer (in rotated order to spread contention) and matching sends, then
 * waits on all of them. Sizes and offsets are per-peer.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

void spreadout_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                         char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                         MPI_Comm comm) {

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int send_typesize, recv_typesize;
	MPI_Type_size(sendtype, &send_typesize);
	MPI_Type_size(recvtype, &recv_typesize);

	MPI_Request *req  = (MPI_Request *) malloc(2 * nprocs * sizeof(MPI_Request));
	MPI_Status  *stat = (MPI_Status  *) malloc(2 * nprocs * sizeof(MPI_Status));

	// Post all recvs first (rotated source order to spread load on a hot peer)
	for (int i = 0; i < nprocs; i++) {
		int src = (rank + i) % nprocs;
		MPI_Irecv(&recvbuf[(size_t)rdispls[src] * recv_typesize],
		          recvcounts[src] * recv_typesize, MPI_CHAR,
		          src, 0, comm, &req[i]);
	}

	// Then post all sends (rotated destination order)
	for (int i = 0; i < nprocs; i++) {
		int dst = (rank - i + nprocs) % nprocs;
		MPI_Isend(&sendbuf[(size_t)sdispls[dst] * send_typesize],
		          sendcounts[dst] * send_typesize, MPI_CHAR,
		          dst, 0, comm, &req[i + nprocs]);
	}

	MPI_Waitall(2 * nprocs, req, stat);
	free(req);
	free(stat);
}
