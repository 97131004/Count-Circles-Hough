#include "globals.h"

namespace globals {
	/*! \brief List of all execution times in nanoseconds: Total, Hough, Hough No-MPI. */
	vector<tuple<long long, long long, long long>> runtimes;
}