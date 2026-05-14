/*
 * bruck_ata.cpp
 *
 * Classic (binary, r=2) Bruck alltoall.
 *
 * Reference: Bruck, Ho, Kipnis, Upfal, Weathersby,
 *            "Efficient Algorithms for All-to-All Communications in
 *             Multiport Message-Passing Systems", IEEE TPDS, 1997.
 *
 * Steps:
 *   1. Local pre-rotation: tmp[i] = sendbuf[(i + rank) % P], so position 0
 *      holds self block, position 1 holds dest=rank+1's block, etc.
 *   2. ceil(log2(P)) rounds. At round k (d = 2^k), send all blocks at
 *      positions whose k-th bit is 1 to peer (rank + d) % P, receive the
 *      same set of positions from peer (rank - d + P) % P.
 *   3. Inverse rotation: recvbuf[(rank - i + P) % P] = tmp[i], so each
 *      slot in recvbuf is indexed by source rank.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

void basic_bruck_alltoall(char *sendbuf, int sendcount, MPI_Datatype sendtype,
                          char *recvbuf, int recvcount, MPI_Datatype recvtype,
                          MPI_Comm comm) {

	int rank, P;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &P);

	int typesize;
	MPI_Type_size(sendtype, &typesize);
	size_t unit_size = (size_t)sendcount * typesize;

	// 1. Local pre-rotation into temp buffer
	char *tmp = (char*) malloc((size_t)P * unit_size);
	for (int i = 0; i < P; i++) {
		memcpy(tmp + (size_t)i * unit_size,
		       sendbuf + (size_t)((i + rank) % P) * unit_size,
		       unit_size);
	}

	// 2. ceil(log2(P)) communication rounds
	int w = (int)ceil(log((double)P) / log(2.0));
	if (w < 1) w = 1;

	// Scratch buffers; sized for the worst case (up to P/2 blocks per round)
	char *send_pack = (char*) malloc((size_t)P * unit_size);
	char *recv_pack = (char*) malloc((size_t)P * unit_size);
	int *positions  = (int*)  malloc((size_t)P * sizeof(int));

	for (int k = 0; k < w; k++) {
		int d = 1 << k;
		int peer_send = (rank + d) % P;
		int peer_recv = (rank - d + P) % P;

		// Pack: positions i in [1, P) with bit k of i set
		int count = 0;
		for (int i = 1; i < P; i++) {
			if (i & d) {
				positions[count] = i;
				memcpy(send_pack + (size_t)count * unit_size,
				       tmp + (size_t)i * unit_size,
				       unit_size);
				count++;
			}
		}

		size_t comm_bytes = (size_t)count * unit_size;
		MPI_Sendrecv(send_pack, comm_bytes, MPI_CHAR, peer_send, 0,
		             recv_pack, comm_bytes, MPI_CHAR, peer_recv, 0,
		             comm, MPI_STATUS_IGNORE);

		// Unpack into the same positions
		for (int i = 0; i < count; i++) {
			memcpy(tmp + (size_t)positions[i] * unit_size,
			       recv_pack + (size_t)i * unit_size,
			       unit_size);
		}
	}

	// 3. Inverse rotation: position i in tmp belongs to source (rank - i + P) % P
	for (int i = 0; i < P; i++) {
		memcpy(recvbuf + (size_t)((rank - i + P) % P) * unit_size,
		       tmp + (size_t)i * unit_size,
		       unit_size);
	}

	free(tmp);
	free(send_pack);
	free(recv_pack);
	free(positions);
}
