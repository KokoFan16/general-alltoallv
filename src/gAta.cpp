/*
 * gAta.cpp
 *
 * Uniform alltoall: every process sends `sendcount` elements to each other
 * process and receives `recvcount` elements from each other process. Derived
 * from gAtav.cpp by removing the per-pair count/displacement bookkeeping and
 * the metadata exchange round (sizes are already known to both sides).
 *
 *      Author: kokofan
 */

#include "gAta.h"

int gata_algorithm (int r, int b, char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { r = 2; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	if ( r > nprocs - 1 ) { r = nprocs - 1; }
	if (b <= 0 || b > nprocs) b = nprocs;

	int w, max_rank, nlpow, d, K, i, num_reqs;
	int rotate_index_array[nprocs];
	w = 0, nlpow = 1, max_rank = nprocs - 1;

	while (max_rank) { w++; max_rank /= r; }
	for (i = 0; i < w - 1; i++) { nlpow *= r; }
	d = (nlpow*r - nprocs) / nlpow;
	K = w * (r - 1) - d;

	int rem1 = K + 1, rem2 = r + 1;
	char *extra_buffer = nullptr, *temp_recv_buffer = nullptr, *temp_send_buffer = nullptr;
	int extra_ids[nprocs - rem2];
	memset(extra_ids, -1, sizeof(extra_ids));
	int spoint = 1, distance = 1, next_distance = distance*r, di = 0;

	const size_t block_size = (size_t)sendcount * typesize;

	// Scratch buffer cached across calls — grow on demand, never shrink.
	// Layout: [extra | temp_recv | temp_send]. Leaks at process exit (intentional).
	static thread_local char  *g_scratch     = nullptr;
	static thread_local size_t g_scratch_cap = 0;

	size_t extra_sz = block_size * (size_t)(nprocs - rem1);
	size_t tsend_sz = block_size * (size_t)nprocs;
	size_t trecv_sz = block_size * (size_t)nprocs;

	if (K < nprocs - 1) {
		// create local index array after rotation
		for (i = 0; i < nprocs; i++) { rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs; }

		size_t want = extra_sz + trecv_sz + tsend_sz;
		if (g_scratch_cap < want) {
			free(g_scratch);
			g_scratch = (char*) malloc(want);
			if (g_scratch == nullptr) {
				g_scratch_cap = 0;
				fprintf(stderr, "buffer allocation failed!\n");
				return 1;
			}
			g_scratch_cap = want;
		}
		extra_buffer     = g_scratch;
		temp_recv_buffer = g_scratch + extra_sz;
		temp_send_buffer = g_scratch + extra_sz + trecv_sz;

		for (int x = 0; x < w; x++) {
			for (int z = 1; z < r; z ++) {
				spoint = z * distance;
				if (spoint > nprocs) { break; }
				int end = (spoint + distance > nprocs)? nprocs : spoint + distance;
				for (i = spoint + 1; i < end; i++) {
					extra_ids[i-rem2] = di++;
				}
			}
			distance *= r;
		}
	}

	// copy data destined for self
	memcpy(&recvbuf[rank * block_size], &sendbuf[rank * block_size], block_size);

	int sent_blocks[r-1][nlpow];
	int nc, rem, ns, ze, ss;
	spoint = 1, distance = 1, next_distance = distance*r;

	MPI_Request* reqs  = (MPI_Request *) malloc(2 * r * sizeof(MPI_Request));
	MPI_Status*  stats = (MPI_Status  *) malloc(2 * r * sizeof(MPI_Status));
	if (reqs == nullptr || stats == nullptr) {
		fprintf(stderr, "MPI_Request/Status allocation failed!\n");
		return 1;
	}

	for (int x = 0; x < w; x++) {
		ze = (x == w - 1)? r - d: r;
		int zoffset = 0, zc = ze-1;
		int zns[zc];

		for (int k = 1; k < ze; k += b) {
			ss = ze - k < b ? ze - k : b;
			num_reqs = 0;
			size_t send_zoffset = 0;

			for (int s = 0; s < ss; s++) {

				int z = k + s;

				spoint = z * distance;
				nc = nprocs / next_distance * distance, rem = nprocs % next_distance - spoint;
				if (rem < 0) { rem = 0; }
				ns = (rem > distance)? (nc + distance) : (nc + rem);
				zns[z-1] = ns;

				int recvrank = (rank + spoint) % nprocs;
				int sendrank = (rank - spoint + nprocs) % nprocs;

				if (ns == 1) {
					MPI_Irecv(&recvbuf[recvrank * block_size], block_size, MPI_CHAR,
							  recvrank, 1, comm, &reqs[num_reqs++]);
					MPI_Isend(&sendbuf[sendrank * block_size], block_size, MPI_CHAR,
							  sendrank, 1, comm, &reqs[num_reqs++]);
				}
				else {
					di = 0;
					for (int i = spoint; i < nprocs; i += next_distance) {
						int j_end = (i+distance > nprocs)? nprocs: i+distance;
						for (int j = i; j < j_end; j++) {
							int id = (j + rank) % nprocs;
							sent_blocks[z-1][di++] = id;
						}
					}

					// prepare send data (block sizes are all uniform; no metadata exchange)
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;
						size_t pos = send_zoffset + (size_t)i * block_size;

						if (i % distance == 0) {
							memcpy(&temp_send_buffer[pos],
								   &sendbuf[send_index * block_size], block_size);
						}
						else {
							memcpy(&temp_send_buffer[pos],
								   &extra_buffer[(size_t)extra_ids[o] * block_size], block_size);
						}
					}

					size_t total = (size_t)di * block_size;
					MPI_Irecv(&temp_recv_buffer[zoffset], total, MPI_CHAR,
							  recvrank, recvrank+z, comm, &reqs[num_reqs++]);
					MPI_Isend(&temp_send_buffer[send_zoffset], total, MPI_CHAR,
							  sendrank, rank+z, comm, &reqs[num_reqs++]);

					zoffset      += total;
					send_zoffset += total;
				}
			}

			MPI_Waitall(num_reqs, reqs, stats);
			for (int i = 0; i < num_reqs; i++) {
				if (stats[i].MPI_ERROR != MPI_SUCCESS) {
					fprintf(stderr, "Request %d encountered an error: %d\n", i, stats[i].MPI_ERROR);
				}
			}
		}

		if (K < nprocs - 1) {
			// place received blocks into final or extra slots
			size_t offset = 0;
			for (int i = 0; i < zc; i++) {
				for (int j = 0; j < zns[i]; j++) {
					if (zns[i] > 1) {
						int o = (sent_blocks[i][j] - rank + nprocs) % nprocs - rem2;
						if (j < distance) {
							memcpy(&recvbuf[sent_blocks[i][j] * block_size],
								   &temp_recv_buffer[offset], block_size);
						}
						else {
							memcpy(&extra_buffer[(size_t)extra_ids[o] * block_size],
								   &temp_recv_buffer[offset], block_size);
						}
						offset += block_size;
					}
				}
			}
		}

		distance *= r;
		next_distance *= r;
	}

	// g_scratch is intentionally kept allocated across calls.
	free(reqs);
	free(stats);

	return 0;
}
