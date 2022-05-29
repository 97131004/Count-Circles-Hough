#pragma once

#include "globals.h"

/*!
 * \brief Collection of edge detection algorithms.
 * \copyright MIT License
 * \author 97131004
 */
class edges
{
public:
	static Mat canny(Mat& src, const int& tresh1, const int& tresh2, const int& ksize = 3);
	static Mat sobel(Mat& src, const int& tresh_bw, const int& ksize = 3);
};

