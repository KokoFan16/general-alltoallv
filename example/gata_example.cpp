/*
 * gata_example.cpp
 *
 *  Driver for gata_algorithm (uniform alltoall).
 */

#include "../src/gAta.h"

static int rank, nprocs;
static void run_gata(int loopcount, int nprocs, std::vector<int> bases);

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


static void run_gata(int loopcount, int nprocs, std::vector<int> bases) {

	int mpi_errno = MPI_SUCCESS;
	int basecount = bases.size();
	for (int n = 2; n <= 1024; n = n * 2) {

		// Uniform: every pair exchanges exactly `n` long-longs
		long long* send_buffer = new long long[(size_t)nprocs * n];
		long long* recv_buffer = new long long[(size_t)nprocs * n];

		int index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < n; j++)
				send_buffer[index++] = i + rank * 10;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < basecount; i++) {
			int b = 2;
			for (int it = 0; it < loopcount; it++) {
				double st = MPI_Wtime();
				mpi_errno = gata_algorithm(bases[i], b,
						(char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG,
						(char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG,
						MPI_COMM_WORLD);
				double et = MPI_Wtime();
				double total_time = et - st;

				if (mpi_errno != MPI_SUCCESS)
					std::cout << "gata_algorithm fail!" << std::endl;

				// correctness: recv_buffer[i*n + j] should hold (rank + i*10)
				int error = 0;
				for (int p = 0; p < nprocs; p++) {
					for (int s = 0; s < n; s++) {
						if ((recv_buffer[p*n + s] % 10) != (rank % 10)) error++;
						if (p != 0 && recv_buffer[p*n + s] == 0) error++;
					}
				}
				if (error > 0) {
					std::cout << rank << " " << n << " " << b
					          << " [gata] base " << bases[i] << " has errors" << std::endl;
				}

				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time) {
					std::cout << "[gata] " << nprocs << ", " << n << ", " << b
					          << ", " << bases[i] << ", " << max_time << std::endl;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		delete[] send_buffer;
		delete[] recv_buffer;
	}
}
