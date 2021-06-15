#include "blur.h"

/*!
 * \brief Performing gaussian blur on an image.
 * \param src Input image
 * \param ksize Kernel size (must be odd)
 */
Mat blur::gaussian(Mat& src, const int& ksize)
{
	Mat blurred;
	GaussianBlur(src, blurred, Size(ksize, ksize), 0, 0, BORDER_DEFAULT);
	return blurred;
}

/*!
 * \brief Performing median filter on an image.
 * \param src Input image
 * \param ksize Kernel size  (must be odd)
 */
Mat blur::median(Mat& src, const int& ksize)
{
	Mat blurred;
	cv::medianBlur(src, blurred, ksize);
	return blurred;
}
