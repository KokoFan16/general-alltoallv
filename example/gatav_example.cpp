/*
 * gatav_example.cpp
 *
 *  Driver for tuna2_algorithm (alltoallv) plus reference implementations
 *  from MPICH and OpenMPI, all timed and compared head-to-head. Sweeps
 *  radix r and b (powers of 2 plus r-1).
 */

#include "../src/gAta.h"

static int rank, nprocs;
static void run_gatav(int loopcount, int nprocs, std::vector<int> bases);

int main(int argc, char **argv) {
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
		std::cout << "ERROR: MPI_Init error\n" << std::endl;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc < 3) {
		if (rank == 0)
			std::cout << "Usage: mpirun -n <nprocs> " << argv[0]
			          << " <loop-count> <base-list>" << std::endl;
		MPI_Finalize();
		return -1;
	}

	int loopCount = atoi(argv[1]);
	std::vector<int> bases;
	for (int i = 2; i < argc; i++)
		bases.push_back(atoi(argv[i]));

	run_gatav(loopCount, nprocs, bases);

	MPI_Finalize();
	return 0;
}

// Restore sendbuf to known pattern: each block i contains (i + rank*10).
static void reset_sendbuf(long long *send, int *sendcounts, int *sdispls,
                          int rank, int nprocs) {
	for (int i = 0; i < nprocs; i++) {
		long long v = i + (long long)rank * 10;
		for (int j = 0; j < sendcounts[i]; j++)
			send[sdispls[i] + j] = v;
	}
}

// Time `fn`, reduce max across ranks, log with `tag`. Reset sendbuf before
// each call so an algorithm that scratches sendbuf doesn't poison the next.
template<typename Fn>
static void bench(const char *tag, int n, int nprocs, int extra1, int extra2,
                  long long *send, int *sendcounts, int *sdispls,
                  int *recvcounts, long long *recv, Fn &&fn) {
	reset_sendbuf(send, sendcounts, sdispls, rank, nprocs);
	MPI_Barrier(MPI_COMM_WORLD);
	double st = MPI_Wtime();
	fn();
	double et = MPI_Wtime();
	double total_time = et - st;

	int error = check_errors(recvcounts, recv, rank, nprocs);
	if (error > 0)
		std::cout << "[" << tag << "] rank " << rank << " n=" << n << " has errors\n";

	double max_time = 0;
	MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		std::cout << "[" << tag << "] " << nprocs << ", " << n
		          << ", " << extra1 << ", " << extra2
		          << ", " << max_time << std::endl;
	}
}


static void run_gatav(int loopcount, int nprocs, std::vector<int> bases) {

	for (int n = 2; n <= 8192; n = n * 2) {

		int sendcounts[nprocs];
		int sdispls[nprocs];
		int soffset = 0;

		// Uniform-random non-uniform distribution: each rank picks its own
		// (deterministic from rank + n), peer counts ∈ [0, n].
		// n is the MAX per-peer count; average count is n/2 so the total
		// volume is ~half of the uniform-`n` case.
		std::mt19937 rng((unsigned)(rank * 1000003u + (unsigned)n));
		std::uniform_int_distribution<int> dist(0, n);
		for (int i = 0; i < nprocs; i++) sendcounts[i] = dist(rng);

		for (int i = 0; i < nprocs; i++) {
			sdispls[i] = soffset;
			soffset += sendcounts[i];
		}

		int recvcounts[nprocs];
		MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
		int rdispls[nprocs];
		int roffset = 0;
		for (int i = 0; i < nprocs; i++) {
			rdispls[i] = roffset;
			roffset += recvcounts[i];
		}

		long long *send_buffer = new long long[soffset];
		long long *recv_buffer = new long long[roffset];

		MPI_Barrier(MPI_COMM_WORLD);

		for (int it = 0; it < loopcount; it++) {

			// --- system MPI_Alltoallv (baseline) ---
			bench("MPI_Alltoallv", n, nprocs, 0, 0,
			      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
				MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
				              recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
				              MPI_COMM_WORLD);
			});

			// --- OpenMPI basic linear ---
			bench("OMPI-linear", n, nprocs, 0, 0,
			      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
				ompi_alltoallv_intra_basic_linear(
					(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
					MPI_COMM_WORLD);
			});

			// --- OpenMPI pairwise ---
			bench("OMPI-pairwise", n, nprocs, 0, 0,
			      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
				ompi_alltoallv_intra_pairwise(
					(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
					MPI_COMM_WORLD);
			});

			// --- MPICH scattered: sweep b over powers of 2 up to nprocs ---
			{
				std::vector<int> mpich_b_values;
				for (int b = 1; b < nprocs; b *= 2) mpich_b_values.push_back(b);
				mpich_b_values.push_back(nprocs);
				for (int mpich_b : mpich_b_values) {
					bench("MPICH-scattered", n, nprocs, mpich_b, 0,
					      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
						MPICH_intra_scattered(mpich_b,
							(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
							(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
							MPI_COMM_WORLD);
					});
				}
			}

			// --- Spreadout ---
			bench("Spreadout", n, nprocs, 0, 0,
			      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
				spreadout_alltoallv(
					(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
					MPI_COMM_WORLD);
			});

			// --- classic binary Bruck (r=2, no tuning) ---
			bench("Bruck", n, nprocs, 0, 0,
			      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
				basic_bruck_alltoallv(
					(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
					MPI_COMM_WORLD);
			});

			// --- tuna2_algorithm (gAtav, your alltoallv) ---
			// Sweep r over user-provided bases; for each r sweep b over
			// powers of 2 up to r-1, plus r-1.
			for (size_t i = 0; i < bases.size(); i++) {
				int r = bases[i];

				std::vector<int> b_values;
				for (int b = 1; b < r; b *= 2) b_values.push_back(b);
				if (r - 1 >= 1 && (b_values.empty() || b_values.back() != r - 1))
					b_values.push_back(r - 1);

				for (int b : b_values) {
					bench("gatav", n, nprocs, b, r,
					      send_buffer, sendcounts, sdispls, recvcounts, recv_buffer, [&]{
						tuna2_algorithm(r, b,
							(char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
							(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG,
							MPI_COMM_WORLD);
					});
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		delete[] send_buffer;
		delete[] recv_buffer;
	}
}
