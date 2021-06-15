#pragma once

#include "globals.h"

/*!
 * \brief Performs hough transform algorithm.
 * \copyright MIT License
 * \author notpavlov
 */
class hough
{
private:
	static void fill_img_into_2d_array(uchar* arr, const Mat& img, const int& width, const int& height);
	static void ind_3d_to_1d(int& ind, const int& x, const int& y, const int& z, const int& width, const int& height);
	static void ind_1d_to_3d(int ind, int& x, int& y, int& z, const int& width, const int& height);
	static void ind_2d_to_1d(int& ind, const int& x, const int& y, const int& width);

public:
	static Mat circle(
		ImpType imp_type, 
		MpiType mpi_type,
		Mat& src,
		Mat& src_image,
		const int& min_radius,
		const int& max_radius,
		const int& peak_tresh,
		const bool& use_binning,
		const int& bin_size,
		const bool& use_spacing,
		const int& spacing_size,
		const int& world_size,
		const int& world_rank,
		const int& omp_threads);
};

