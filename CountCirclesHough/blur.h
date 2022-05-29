#pragma once

#include "globals.h"

/*!
 * \brief Collection of blur filtering functions.
 * \copyright MIT License
 * \author 97131004
 */
class blur
{
public:
	static Mat gaussian(Mat& src, const int& ksize = 3);
	static Mat median(Mat& src, const int& ksize = 5);
};

