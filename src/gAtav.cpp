/*
 * gAtav.cpp
 *      Author: kokofan
 */

#include "gAta.h"

int tuna2_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { r = 2; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	if ( r > nprocs - 1 ) { r = nprocs - 1; }
	if (b <= 0 || b > nprocs) b = nprocs;

	int w, max_rank, nlpow, d, K, i, num_reqs;
	int local_max_count=0, max_send_count=0;
	int rotate_index_array[nprocs];
	w = 0, nlpow = 1, max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }   // number of bits required of r representation
    for (i = 0; i < w - 1; i++) { nlpow *= r; }   // maximum send number of elements
	d = (nlpow*r - nprocs) / nlpow; // calculate the number of highest digits
	K = w * (r - 1) - d; // the total number of communication rounds

	int rem1 = K + 1, rem2 = r + 1;
	int sendNcopy[nprocs - rem1];
	char *extra_buffer = nullptr, *temp_recv_buffer = nullptr, *temp_send_buffer = nullptr;
	int extra_ids[nprocs - rem2];
	memset(extra_ids, -1, sizeof(extra_ids));
	int spoint = 1, distance = 1, next_distance = distance*r, di = 0;

	// Scratch buffer cached across calls — grow on demand, never shrink.
	static thread_local char  *g_scratch     = nullptr;
	static thread_local size_t g_scratch_cap = 0;

	if (K < nprocs - 1) {
		// 1. Find max send count
		for (i = 0; i < nprocs; i++) {
			if (sendcounts[i] > local_max_count) { local_max_count = sendcounts[i]; }
		}
		MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

		// 2. create local index array after rotation
		for (i = 0; i < nprocs; i++) { rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs; }

		// 3. cached scratch — layout: [extra | temp_recv | temp_send]
		size_t unit     = (size_t)max_send_count * typesize;
		size_t extra_sz = unit * (size_t)(nprocs - rem1);
		size_t trecv_sz = unit * (size_t)nprocs;
		size_t tsend_sz = unit * (size_t)nprocs;
		size_t want     = extra_sz + trecv_sz + tsend_sz;
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

	// copy data that need to be sent to each rank itself
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);


	int sent_blocks[r-1][nlpow];
	int metadata_recv[r-1][nlpow];
	int nc, rem, ns, ze, ss;
	spoint = 1, distance = 1, next_distance = distance*r;

	MPI_Request* reqs = (MPI_Request *) malloc(2 * r * sizeof(MPI_Request));
	if (reqs == nullptr) {
		fprintf(stderr, "MPI_Requests allocation failed!\n");
		return 1;
	}

	MPI_Status* stats = (MPI_Status *) malloc(2 * r * sizeof(MPI_Status));
	if (stats == nullptr) {
		fprintf(stderr, "MPI_Status allocation failed!\n");
		return 1;
	}

	// metadata_send: per-z slot so multiple z's can be in flight concurrently
	int metadata_send[r - 1][nlpow];
	// extra scratch needed for the metadata-batching pass
	MPI_Request md_reqs[2 * r];
	int md_di    [r - 1];
	int md_sendrk[r - 1];
	int md_recvrk[r - 1];

    int comm_size[r-1];
	for (int x = 0; x < w; x++) {
		ze = (x == w - 1)? r - d: r;
		int zoffset = 0, zc = ze-1;
		int zns[zc];

		for (int k = 1; k < ze; k += b) {
			ss = ze - k < b ? ze - k : b;
			num_reqs = 0;
			size_t send_zoffset = 0;
			int num_md = 0;

			// ---- Phase A0: compute, build sent_blocks, build metadata_send ----
			for (int s = 0; s < ss; s++) {
				int z = k + s;

				spoint = z * distance;
				nc = nprocs / next_distance * distance;
				rem = nprocs % next_distance - spoint;
				if (rem < 0) rem = 0;
				ns = (rem > distance)? (nc + distance) : (nc + rem);
				zns[z-1] = ns;

				md_recvrk[z-1] = (rank + spoint) % nprocs;
				md_sendrk[z-1] = (rank - spoint + nprocs) % nprocs;

				if (ns == 1) {
					md_di[z-1] = 0;   // direct exchange, no metadata
				} else {
					di = 0;
					for (int i = spoint; i < nprocs; i += next_distance) {
						int j_end = (i+distance > nprocs)? nprocs: i+distance;
						for (int j = i; j < j_end; j++) {
							int id = (j + rank) % nprocs;
							sent_blocks[z-1][di++] = id;
						}
					}
					md_di[z-1] = di;

					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;
						metadata_send[z-1][i] = (i % distance == 0)
						                        ? sendcounts[send_index]
						                        : sendNcopy[extra_ids[o]];
					}
				}
			}

			// ---- Phase A1: post ALL Irecvs (ns==1 data + ns>1 metadata) ----
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				if (md_di[z-1] == 0) {
					MPI_Irecv(&recvbuf[(size_t)rdispls[md_recvrk[z-1]]*typesize],
					          recvcounts[md_recvrk[z-1]]*typesize, MPI_CHAR,
					          md_recvrk[z-1], 1, comm, &reqs[num_reqs++]);
				} else {
					MPI_Irecv(metadata_recv[z-1], md_di[z-1], MPI_INT,
					          md_recvrk[z-1], 0, comm, &md_reqs[num_md++]);
				}
			}

			// ---- Phase A2: post ALL Isends (ns==1 data + ns>1 metadata) ----
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				if (md_di[z-1] == 0) {
					MPI_Isend(&sendbuf[(size_t)sdispls[md_sendrk[z-1]]*typesize],
					          sendcounts[md_sendrk[z-1]]*typesize, MPI_CHAR,
					          md_sendrk[z-1], 1, comm, &reqs[num_reqs++]);
				} else {
					MPI_Isend(metadata_send[z-1], md_di[z-1], MPI_INT,
					          md_sendrk[z-1], 0, comm, &md_reqs[num_md++]);
				}
			}

			if (num_md > 0) MPI_Waitall(num_md, md_reqs, MPI_STATUSES_IGNORE);

			// ---- Phase B0: pack data using metadata_recv ----
			size_t s_send_off[ss], s_recv_off[ss], s_send_bytes[ss], s_recv_bytes[ss];
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				int di_z = md_di[z-1];
				if (di_z == 0) continue;

				size_t sendCount = 0;
				for (int i = 0; i < di_z; i++)
					sendCount += (size_t)metadata_recv[z-1][i];
				comm_size[z-1] = (int)sendCount;

				size_t offset = 0;
				for (int i = 0; i < di_z; i++) {
					int send_index = rotate_index_array[sent_blocks[z-1][i]];
					int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;
					size_t size;

					if (i % distance == 0) {
						size = (size_t)sendcounts[send_index] * typesize;
						memcpy(&temp_send_buffer[send_zoffset + offset],
						       &sendbuf[(size_t)sdispls[send_index] * typesize], size);
					} else {
						size = (size_t)sendNcopy[extra_ids[o]] * typesize;
						memcpy(&temp_send_buffer[send_zoffset + offset],
						       &extra_buffer[(size_t)extra_ids[o] * max_send_count * typesize], size);
					}
					offset += size;
				}

				size_t recv_bytes = sendCount * typesize;
				s_send_off[s]    = send_zoffset;
				s_recv_off[s]    = zoffset;
				s_send_bytes[s]  = offset;
				s_recv_bytes[s]  = recv_bytes;
				zoffset         += recv_bytes;
				send_zoffset    += offset;
			}

			// ---- Phase B1: post ALL data Irecvs (ns>1 only) ----
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				if (md_di[z-1] == 0) continue;
				MPI_Irecv(&temp_recv_buffer[s_recv_off[s]], (int)s_recv_bytes[s], MPI_CHAR,
				          md_recvrk[z-1], md_recvrk[z-1]+z, comm, &reqs[num_reqs++]);
			}

			// ---- Phase B2: post ALL data Isends (ns>1 only) ----
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				if (md_di[z-1] == 0) continue;
				MPI_Isend(&temp_send_buffer[s_send_off[s]], (int)s_send_bytes[s], MPI_CHAR,
				          md_sendrk[z-1], rank+z, comm, &reqs[num_reqs++]);
			}

			MPI_Waitall(num_reqs, reqs, stats);
			for (int i = 0; i < num_reqs; i++) {
				if (stats[i].MPI_ERROR != MPI_SUCCESS) {
					fprintf(stderr, "Request %d encountered an error: %d\n", i, stats[i].MPI_ERROR);
				}
			}
		}

		if (K < nprocs - 1) {
			// replaces
			size_t offset = 0;
			for (int i = 0; i < zc; i++) {
				for (int j = 0; j < zns[i]; j++) {
					if (zns[i] > 1) {
						size_t size = (size_t)metadata_recv[i][j] * typesize;
						int o = (sent_blocks[i][j] - rank + nprocs) % nprocs - rem2;

						if (j < distance) {
							memcpy(&recvbuf[(size_t)rdispls[sent_blocks[i][j]] * typesize],
							       &temp_recv_buffer[offset], size);
						}
						else {
							memcpy(&extra_buffer[(size_t)extra_ids[o] * max_send_count * typesize],
							       &temp_recv_buffer[offset], size);
							sendNcopy[extra_ids[o]] = metadata_recv[i][j];
						}
						offset += size;
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

