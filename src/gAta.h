/*
 * gAta.h
 *      Author: kokofan
 */

#ifndef SRC_GATA_H_
#define SRC_GATA_H_

#include "../gata_common.h"

int tuna2_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int tuna2_gpu_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int gata_algorithm (int r, int b, char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int gata_gpu_algorithm (int r, int b, char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

// benchmark reference algorithms (non-uniform alltoallv)
int ompi_alltoallv_intra_basic_linear(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int ompi_alltoallv_intra_pairwise(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPICH_intra_scattered(int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

void spreadout_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int exclisive_or_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

// benchmark reference algorithms (uniform alltoall)
int ompi_alltoall_intra_basic_linear(char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int ompi_alltoall_intra_pairwise(char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int MPICH_intra_scattered_ata(int b, char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

void spreadout_alltoall(char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

void basic_bruck_alltoall(char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

#endif /* SRC_GATA_H_ */
