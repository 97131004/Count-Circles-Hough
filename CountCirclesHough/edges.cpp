#include "edges.h"

/*!
 * \brief Performing canny edge detection algorithm on an image.
 * \param src Input image
 * \param tresh1 First threshold for the hysteresis procedure
 * \param tresh2 Second threshold for the hysteresis procedure
 * \param ksize Kernel size (must be odd)
 */
Mat edges::canny(Mat& src, const int& tresh1, const int& tresh2, const int& ksize) {
	Mat dst;
	Canny(src, dst, tresh1, tresh2, ksize);
	return dst;
}

/*!
 * \brief Applying sobel edge detection filter on an image.
 * \param src Input image
 * \param tresh_bw Treshold for black/white (binary) image generation
 * \param ksize Kernel size (must be -1, 1, 3, 5 or 7)
 */
Mat edges::sobel(Mat& src, const int& tresh_bw, const int& ksize) {

	Mat src_gray;
	Mat grad;

	//declare matrices for horizontal and vertical gradients
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	src.copyTo(src_gray);

	//creating horizontal gradient
	Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	//creating vertical gradient
	Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	//merge both gradients (approximately) into a final matrix
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//grad.convertTo(grad, CV_8UC1);

	//convert final image into a binary image
	cv::threshold(grad, grad, tresh_bw, 255.0, THRESH_BINARY);

	return grad;
}