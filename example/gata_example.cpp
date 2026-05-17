/*
 * gata_example.cpp
 *
 *  Driver for gata_algorithm (uniform alltoall) plus reference implementations
 *  from MPICH and OpenMPI, all timed and compared head-to-head.
 */

#include "../src/gAta.h"

static int rank, nprocs;
static void run_gata(int loopcount, int nprocs, std::vector<int> bases);

// Verify recv buffer for a uniform alltoall.
// Sender p writes (this_rank + p*10) into our slot p.
static int check_uniform(long long *recv_buffer, int n, int rank, int nprocs) {
	int error = 0;
	for (int p = 0; p < nprocs; p++) {
		for (int s = 0; s < n; s++) {
			if ((recv_buffer[p*n + s] % 10) != (rank % 10)) error++;
			if (p != 0 && recv_buffer[p*n + s] == 0) error++;
		}
	}
	return error;
}

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

	run_gata(loopCount, nprocs, bases);

	MPI_Finalize();
	return 0;
}

// Refill sendbuf with `i + rank*10` so each algorithm starts from clean state.
// Some implementations (e.g. rbruck_ata) reuse sendbuf as scratch.
static void reset_sendbuf(long long *send, int n, int rank, int nprocs) {
	int idx = 0;
	for (int i = 0; i < nprocs; i++)
		for (int j = 0; j < n; j++)
			send[idx++] = i + rank * 10;
}

// Time `fn` (no args), reduce max across ranks, log result with `tag`.
template<typename Fn>
static void bench(const char *tag, int n, int nprocs, int extra1, int extra2,
                  long long *send, long long *recv, Fn &&fn) {
	reset_sendbuf(send, n, rank, nprocs);
	MPI_Barrier(MPI_COMM_WORLD);
	double st = MPI_Wtime();
	fn();
	double et = MPI_Wtime();
	double total_time = et - st;

	int error = check_uniform(recv, n, rank, nprocs);
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


static void run_gata(int loopcount, int nprocs, std::vector<int> bases) {

	for (int n = 2048; n <= 8192; n = n * 2) {

		long long *send_buffer = new long long[(size_t)nprocs * n];
		long long *recv_buffer = new long long[(size_t)nprocs * n];

		int index = 0;
		for (int i = 0; i < nprocs; i++)
			for (int j = 0; j < n; j++)
				send_buffer[index++] = i + rank * 10;

		MPI_Barrier(MPI_COMM_WORLD);

		for (int it = 0; it < loopcount; it++) {

			// --- system MPI_Alltoall (baseline) ---
			bench("MPI_Alltoall", n, nprocs, 0, 0, send_buffer, recv_buffer, [&]{
				MPI_Alltoall(send_buffer, n, MPI_UNSIGNED_LONG_LONG,
				             recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			});

			// --- OpenMPI basic linear ---
			bench("OMPI-linear", n, nprocs, 0, 0, send_buffer, recv_buffer, [&]{
				ompi_alltoall_intra_basic_linear(
					(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			});

			// --- OpenMPI pairwise ---
			bench("OMPI-pairwise", n, nprocs, 0, 0, send_buffer, recv_buffer, [&]{
				ompi_alltoall_intra_pairwise(
					(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			});

			// --- MPICH scattered: sweep b over powers of 2 up to nprocs ---
			{
				std::vector<int> mpich_b_values;
				for (int b = 1; b < nprocs; b *= 2) mpich_b_values.push_back(b);
				mpich_b_values.push_back(nprocs);
				for (int mpich_b : mpich_b_values) {
					bench("MPICH-scattered", n, nprocs, mpich_b, 0, send_buffer, recv_buffer, [&]{
						MPICH_intra_scattered_ata(mpich_b,
							(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
							(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					});
				}
			}

			// --- Spread-out ---
			bench("Spreadout", n, nprocs, 0, 0, send_buffer, recv_buffer, [&]{
				spreadout_alltoall(
					(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			});

			// --- classic binary Bruck (r=2, no tuning) ---
			bench("Bruck", n, nprocs, 0, 0, send_buffer, recv_buffer, [&]{
				basic_bruck_alltoall(
					(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			});

			// --- gata_algorithm (your implementation) ---
			// Sweep r (radix) over user-provided bases. For each r, sweep b
			// over powers of 2 up to r-1, plus r-1 itself. This keeps
			// the sweep at O(log r) so r=P=8192 only runs ~14 b values instead
			// of 8191.
			for (size_t i = 0; i < bases.size(); i++) {
				int r = bases[i];

				std::vector<int> b_values;
				for (int b = 1; b < r; b *= 2) b_values.push_back(b);
				if (r - 1 >= 1 && (b_values.empty() || b_values.back() != r - 1))
					b_values.push_back(r - 1);

				for (int b : b_values) {
					bench("gata", n, nprocs, b, r, send_buffer, recv_buffer, [&]{
						gata_algorithm(r, b,
							(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
							(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					});
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		delete[] send_buffer;
		delete[] recv_buffer;
	}
}
