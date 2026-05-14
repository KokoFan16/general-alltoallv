/*
 * gata_common.h
 *
 * Common includes for the project.
 *      Author: kokofan
 */

#ifndef GATA_COMMON_H_
#define GATA_COMMON_H_

#include <mpi.h>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <string>
#include <cmath>
#include <vector>

inline int myPow(int x, unsigned int p) {
    if (p == 0) return 1;
    if (p == 1) return x;
    int tmp = myPow(x, p / 2);
    if (p % 2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

inline int check_errors(int *recvcounts, long long *recv_buffer, int rank, int nprocs) {
    int error = 0, index = 0;
    for (int p = 0; p < nprocs; p++) {
        for (int s = 0; s < recvcounts[p]; s++) {
            if (p != 0 || rank != 0) {
                if (recv_buffer[index] == 0) error++;
            }
            if ((recv_buffer[index] % 10) != (rank % 10)) error++;
            index++;
        }
    }
    return error;
}

#endif /* GATA_COMMON_H_ */
