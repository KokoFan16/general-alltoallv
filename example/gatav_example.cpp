/*
 * gatav_example.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
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

		int sendcounts[nprocs];
		memset(sendcounts, 0, nprocs*sizeof(int));
		int sdispls[nprocs];
		int soffset = 0;

		srand(time(NULL));
		for (int i = 0; i < nprocs; i++) sendcounts[i] = n;

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));

		for (int i = 0; i < nprocs; ++i) {
			sdispls[i] = soffset;
			soffset += sendcounts[i];
		}

		int recvcounts[nprocs];
		MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
		int rdispls[nprocs];
		int roffset = 0;
		for (int i = 0; i < nprocs; ++i) {
			rdispls[i] = roffset;
			roffset += recvcounts[i];
		}

		long long* send_buffer = new long long[soffset];
		long long* recv_buffer = new long long[roffset];

		int index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < sendcounts[i]; j++)
				send_buffer[index++] = i + rank * 10;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < basecount; i++) {
			int b = 2;
			for (int it = 0; it < loopcount; it++) {
				double st = MPI_Wtime();
				mpi_errno = tuna2_algorithm(bases[i], b, (char*)send_buffer, sendcounts, sdispls,
						MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls,
						MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				double et = MPI_Wtime();
				double total_time = et - st;

				if (mpi_errno != MPI_SUCCESS)
					std::cout << "tuna2_algorithm fail!" << std::endl;

				int error = check_errors(recvcounts, recv_buffer, rank, nprocs);
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
