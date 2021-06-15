#pragma once

#include <iostream>
#include <cstdio>
#include <chrono>
#include <exception>
#include <thread>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include "mpi.h"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

/*! \brief Implementation type to run. */
enum ImpType { 
	sequential, /**< Sequential execution with no parallization */ 
	openmp, /**< Parallelization with OpenMP */ 
	openmpi  /**< Parallelization with OpenMPI */ 
};

/*! \brief MPI field size to send and receive. */
enum MpiType {
	full, /**< Send full-sized image and receive full-sized accumulator matrix */
	crop /**< Send cropped image and receive cropped accumulator matrix */ 
};

/*! \brief Blur filter type to apply to the image. */
enum BlurType { 
	median, /**< Median Filter */
	gaussian /**< Gaussian Blur */
};

/*! \brief Edge detection algorithm to run on the image. */
enum EdgesType { 
	sobel, /**< Sobel Filter */
	canny /**< Canny Edge Detector */
};

/*! \brief Globally-accessible fields. */
namespace globals {
	extern vector<tuple<long long, long long, long long>> runtimes;
}
