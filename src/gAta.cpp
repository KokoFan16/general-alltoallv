/*
 * gAta.cpp
 *
 * Uniform alltoall under the (r, b) parameter family:
 *   r — Bruck radix
 *   b — in-flight batch size (max concurrent z-steps per Waitall)
 *
 * Implementation: classical Bruck buffer layout (in-place `tmp`) generalised
 * to radix r and in-flight batch size b. The pre-rotation moves sendbuf
 * into a single rotated buffer `tmp`; all forwarded blocks live in `tmp`
 * at fixed positions across rounds. Each sub-batch packs the active
 * positions into a contiguous send-staging buffer, exchanges with the
 * peers, and unpacks the received bytes back to the same positions in
 * `tmp`. A final post-rotation copies `tmp` into recvbuf indexed by
 * source rank.
 *
 *      Author: kokofan
 */

#include "gAta.h"

int gata_algorithm (int r, int b, char *sendbuf, int sendcount, MPI_Datatype sendtype,
		char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	const size_t block_size = (size_t)sendcount * typesize;
	const size_t buf_bytes  = block_size * (size_t)nprocs;

	// ---- Degenerate case: single process -------------------------------
	if (nprocs <= 1) {
		if (nprocs == 1 && block_size > 0)
			memcpy(recvbuf, sendbuf, block_size);
		return 0;
	}

	// ---- Parameter sanitisation ----------------------------------------
	if (r < 2)             r = 2;
	if (r > nprocs)        r = nprocs;     // r = P is valid (pairwise / linear)
	if (b <= 0 || b > nprocs) b = nprocs;

	// ---- Fast path: no-forwarding regime (r >= P-1) --------------------
	// When r >= P-1 every non-zero position in [1, P) has at most one
	// non-zero base-r digit, so each block is sent exactly once and the
	// rotated `tmp` of the general path is a redundant permutation of
	// `sendbuf`. Skip rotation entirely and exchange directly between
	// `sendbuf` and `recvbuf`, still respecting the batch size `b`
	// (sub-batch count = ceil((P-1)/b)).  This collapses to MPICH
	// scattered (b<P-1) or basic linear (b>=P-1) at zero rotation cost.
	if (r >= nprocs - 1) {
		// Self block
		memcpy(&recvbuf[(size_t)rank * block_size],
		       &sendbuf[(size_t)rank * block_size], block_size);

		int max_in_flight = (b > nprocs - 1) ? nprocs - 1 : b;
		MPI_Request *reqs  = (MPI_Request *) malloc(2 * max_in_flight * sizeof(MPI_Request));
		MPI_Status  *stats = (MPI_Status  *) malloc(2 * max_in_flight * sizeof(MPI_Status));
		if (reqs == nullptr || stats == nullptr) {
			fprintf(stderr, "gata: MPI request allocation failed (pairwise path)\n");
			free(reqs); free(stats);
			return 1;
		}

		for (int k = 1; k < nprocs; k += b) {
			int ss = (nprocs - k < b) ? (nprocs - k) : b;
			int num_reqs = 0;

			// Phase 1: post ALL Irecv first (CXI match-list friendly)
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				int recv_peer = (rank - z + nprocs) % nprocs;
				MPI_Irecv(&recvbuf[(size_t)recv_peer * block_size],
				          block_size, MPI_CHAR,
				          recv_peer, recv_peer + z, comm,
				          &reqs[num_reqs++]);
			}
			// Phase 2: post ALL Isend
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				int send_peer = (rank + z) % nprocs;
				MPI_Isend(&sendbuf[(size_t)send_peer * block_size],
				          block_size, MPI_CHAR,
				          send_peer, rank + z, comm,
				          &reqs[num_reqs++]);
			}

			MPI_Waitall(num_reqs, reqs, stats);
			for (int i = 0; i < num_reqs; i++) {
				if (stats[i].MPI_ERROR != MPI_SUCCESS) {
					fprintf(stderr, "gata: request %d error %d (pairwise path)\n",
					        i, stats[i].MPI_ERROR);
				}
			}
		}

		free(reqs);
		free(stats);
		return 0;
	}

	// ---- Algorithm metadata --------------------------------------------
	// w      = number of base-r digits needed for indices 0..nprocs-1
	// nlpow  = r^(w-1), place value of the top digit
	// d      = top-digit deficit when P is not a perfect power of r
	int w = 0, max_rank = nprocs - 1, nlpow = 1;
	while (max_rank) { w++; max_rank /= r; }
	for (int i = 0; i < w - 1; i++) nlpow *= r;
	int d = (nlpow * r - nprocs) / nlpow;

	// ---- Scratch buffer ------------------------------------------------
	// Layout:  [ tmp | temp_send | temp_recv ], each P * block_size.
	// Cached across calls (grow-only, intentionally leaked at exit).
	static thread_local char  *g_scratch     = nullptr;
	static thread_local size_t g_scratch_cap = 0;

	size_t want = buf_bytes * 3;
	if (g_scratch_cap < want) {
		free(g_scratch);
		g_scratch = (char *) malloc(want);
		if (g_scratch == nullptr) {
			g_scratch_cap = 0;
			fprintf(stderr, "gata: scratch allocation of %zu bytes failed\n", want);
			return 1;
		}
		g_scratch_cap = want;
	}
	char *tmp       = g_scratch;
	char *temp_send = g_scratch + buf_bytes;
	char *temp_recv = g_scratch + buf_bytes * 2;

	// ---- Pre-rotation:  tmp[i] = sendbuf[(i + rank) % P] ---------------
	// Position i in tmp now holds rank's data destined for (i+rank)%P.
	// Position 0 is the self block; it is never touched by any z-step
	// (digit-x of 0 is 0 for every x), so the self-block is delivered
	// automatically by the post-rotation.
	for (int i = 0; i < nprocs; i++) {
		memcpy(tmp + (size_t)i * block_size,
		       sendbuf + (size_t)((i + rank) % nprocs) * block_size,
		       block_size);
	}

	// ---- MPI request scratch -------------------------------------------
	// At most 2(r-1) in flight per sub-batch (one Irecv + one Isend per z).
	int max_reqs = 2 * r;
	MPI_Request *reqs  = (MPI_Request *) malloc(max_reqs * sizeof(MPI_Request));
	MPI_Status  *stats = (MPI_Status  *) malloc(max_reqs * sizeof(MPI_Status));
	if (reqs == nullptr || stats == nullptr) {
		fprintf(stderr, "gata: MPI request allocation failed\n");
		free(reqs); free(stats);
		return 1;
	}

	// ---- Main loop:  w rounds, ceil((r-1)/b) sub-batches each ----------
	int distance = 1;
	for (int x = 0; x < w; x++) {
		int next_distance = distance * r;
		int ze = (x == w - 1) ? r - d : r;   // last-round deficit

		for (int k = 1; k < ze; k += b) {
			int ss = (ze - k < b) ? (ze - k) : b;

			// Per-z bookkeeping kept around for sub-phases 1/2/3.
			int    s_send_peer[ss], s_recv_peer[ss];
			size_t s_off[ss];   // start of this z's region in temp_send/temp_recv
			size_t s_bytes[ss]; // number of bytes for this z

			// ---- Sub-phase 0: enumerate positions and pack -------------
			// For z-step z in round x, the active positions are exactly
			// those whose base-r digit at place x equals z, i.e.
			//   p in [z*distance, z*distance + distance) ∪
			//        [z*distance +  next_distance, ...) ∪ ...
			// clipped to [0, nprocs).  We pack tmp[p] into a contiguous
			// region of temp_send in enumeration order.
			size_t send_zoffset = 0;
			for (int s = 0; s < ss; s++) {
				int z = k + s;
				int spoint = z * distance;

				s_send_peer[s] = (rank + spoint) % nprocs;
				s_recv_peer[s] = (rank - spoint + nprocs) % nprocs;
				s_off[s]       = send_zoffset;

				int np = 0;
				for (int i = spoint; i < nprocs; i += next_distance) {
					int j_end = (i + distance < nprocs) ? (i + distance) : nprocs;
					for (int j = i; j < j_end; j++) {
						memcpy(temp_send + send_zoffset + (size_t)np * block_size,
						       tmp + (size_t)j * block_size,
						       block_size);
						np++;
					}
				}
				s_bytes[s]    = (size_t)np * block_size;
				send_zoffset += s_bytes[s];
			}

			// ---- Sub-phase 1: post ALL Irecv first ---------------------
			// On CXI / Slingshot the hardware match list works best when
			// receive buffers are pre-posted before any send arrives.
			int num_reqs = 0;
			for (int s = 0; s < ss; s++) {
				if (s_bytes[s] == 0) continue;
				int z = k + s;
				MPI_Irecv(temp_recv + s_off[s], s_bytes[s], MPI_CHAR,
				          s_recv_peer[s], s_recv_peer[s] + z, comm,
				          &reqs[num_reqs++]);
			}
			// ---- Sub-phase 2: post ALL Isend ---------------------------
			for (int s = 0; s < ss; s++) {
				if (s_bytes[s] == 0) continue;
				int z = k + s;
				MPI_Isend(temp_send + s_off[s], s_bytes[s], MPI_CHAR,
				          s_send_peer[s], rank + z, comm,
				          &reqs[num_reqs++]);
			}

			MPI_Waitall(num_reqs, reqs, stats);
			for (int i = 0; i < num_reqs; i++) {
				if (stats[i].MPI_ERROR != MPI_SUCCESS) {
					fprintf(stderr, "gata: request %d error %d\n", i, stats[i].MPI_ERROR);
				}
			}

			// ---- Sub-phase 3: unpack temp_recv back into tmp -----------
			// Re-enumerate the same position list and write received
			// blocks to the same tmp positions.  Forwarded blocks now
			// sit at their next-round position; final-destination blocks
			// will be picked up by post-rotation.
			for (int s = 0; s < ss; s++) {
				if (s_bytes[s] == 0) continue;
				int z = k + s;
				int spoint = z * distance;

				size_t off = s_off[s];
				int np = 0;
				for (int i = spoint; i < nprocs; i += next_distance) {
					int j_end = (i + distance < nprocs) ? (i + distance) : nprocs;
					for (int j = i; j < j_end; j++) {
						memcpy(tmp + (size_t)j * block_size,
						       temp_recv + off + (size_t)np * block_size,
						       block_size);
						np++;
					}
				}
			}
		}

		distance = next_distance;
	}

	// ---- Post-rotation:  recvbuf[(rank - i + P) % P] = tmp[i] ----------
	// After all rounds, tmp[i] at rank R holds data from source (R-i)%P
	// (all destined for R).  Inverse rotation places it at recvbuf
	// indexed by source rank.
	for (int i = 0; i < nprocs; i++) {
		memcpy(recvbuf + (size_t)((rank - i + nprocs) % nprocs) * block_size,
		       tmp + (size_t)i * block_size,
		       block_size);
	}

	free(reqs);
	free(stats);
	return 0;
}
