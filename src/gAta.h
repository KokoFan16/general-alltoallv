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

#endif /* SRC_GATA_H_ */
