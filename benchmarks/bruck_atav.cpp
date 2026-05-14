/*
 * bruck_atav.cpp
 *
 * Classic (binary, r=2) Bruck alltoallv for variable per-peer message sizes.
 *
 * Reference: Bruck, Ho, Kipnis, Upfal, Weathersby (IEEE TPDS, 1997), extended
 * to alltoallv with a metadata sendrecv each round so the receiver knows how
 * many bytes to expect.
 *
 *      Author: kokofan
 */

#include "../gata_common.h"

void basic_bruck_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                           char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                           MPI_Comm comm) {

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find the global max send count (so extra_buffer slot size is uniform)
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count) local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	// 2. Rotation index: rotated position i ↔ original slot (2*rank - i + P) % P
	int rotate_index_array[nprocs];
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;

	// 3. Temp buffers
	int max_send_elements = (nprocs + 1) / 2;
	char *extra_buffer     = (char*) malloc((size_t)max_send_count * typesize * nprocs);
	char *temp_send_buffer = (char*) malloc((size_t)max_send_count * typesize * max_send_elements);
	char *temp_recv_buffer = (char*) malloc((size_t)max_send_count * typesize * max_send_elements);

	// pos_status[s] = 0 → block s is still in sendbuf; 1 → it lives in extra_buffer
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs * sizeof(int));

	// sendNcopy tracks the *current* size of block s without clobbering sendcounts[s]
	int sendNcopy[nprocs];
	memcpy(sendNcopy, sendcounts, nprocs * sizeof(int));

	// Self block goes straight to recvbuf
	memcpy(&recvbuf[(size_t)rdispls[rank] * typesize],
	       &sendbuf[(size_t)sdispls[rank] * typesize],
	       (size_t)recvcounts[rank] * typesize);

	for (int k = 1; k < nprocs; k <<= 1) {

		// 1) Which rotated positions have bit k set?
		int send_indexes[max_send_elements];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++) {
			if (i & k)
				send_indexes[sendb_num++] = (rank + i) % nprocs;
		}

		// 2) Pack send buffer + metadata
		int metadata_send[sendb_num];
		size_t offset = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];
			metadata_send[i] = sendNcopy[send_index];
			size_t bytes = (size_t)sendNcopy[send_index] * typesize;
			if (pos_status[send_index] == 0)
				memcpy(&temp_send_buffer[offset],
				       &sendbuf[(size_t)sdispls[send_index] * typesize], bytes);
			else
				memcpy(&temp_send_buffer[offset],
				       &extra_buffer[(size_t)send_indexes[i] * max_send_count * typesize], bytes);
			offset += bytes;
		}

		// 3) Exchange metadata, then data
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		int metadata_recv[sendb_num];
		MPI_Sendrecv(metadata_send, sendb_num, MPI_INT, sendrank, 0,
		             metadata_recv,  sendb_num, MPI_INT, recvrank, 0,
		             comm, MPI_STATUS_IGNORE);

		int sendCount = 0;
		for (int i = 0; i < sendb_num; i++) sendCount += metadata_recv[i];

		MPI_Sendrecv(temp_send_buffer, (int)offset, MPI_CHAR, sendrank, 1,
		             temp_recv_buffer, sendCount * typesize, MPI_CHAR, recvrank, 1,
		             comm, MPI_STATUS_IGNORE);

		// 4) Place received blocks: final destination → recvbuf, otherwise stash
		size_t roff = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];
			size_t bytes = (size_t)metadata_recv[i] * typesize;

			if ((send_indexes[i] - rank + nprocs) % nprocs < (k << 1)) {
				// block has reached its final rotated position
				memcpy(&recvbuf[(size_t)rdispls[send_indexes[i]] * typesize],
				       &temp_recv_buffer[roff], bytes);
			} else {
				// keep forwarding in subsequent rounds
				memcpy(&extra_buffer[(size_t)send_indexes[i] * max_send_count * typesize],
				       &temp_recv_buffer[roff], bytes);
			}
			roff += bytes;
			pos_status[send_index] = 1;
			sendNcopy[send_index] = metadata_recv[i];
		}
	}

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);
}
