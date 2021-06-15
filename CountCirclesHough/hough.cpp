#include "hough.h"

/*!
 * \brief Fills image into a 2D-array (represented as 1D-array).
 * \param arr Destination 2D-array (represnted as 1D-array)
 * \param img Source image
 * \param width Image width
 * \param height Image height
 */
void hough::fill_img_into_2d_array(uchar* arr, const Mat& img, const int& width, const int& height) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			arr[width * i + j] = (uchar)img.at<uchar>(i, j);
		}
	}
}

/*!
 * \brief Converts 2D-array index to 1D-array index.
 * \param ind Output 1D-array index
 * \param x Input 2D-array X-index
 * \param y Input 2D-array Y-index
 * \param width 2D-array width
 */
void hough::ind_2d_to_1d(int& ind, const int& x, const int& y, const int& width) {
	ind = width * y + x;
}

/*!
 * \brief Converts 3D-array index to 1D-array index.
 * \param ind Output 1D-array index
 * \param x Input 3D-array X-index
 * \param y Input 3D-array Y-index
 * \param z Input 3D-array Z-index
 * \param width 3D-array width
 * \param height 3D-array height
 */
void hough::ind_3d_to_1d(int& ind, const int& x, const int& y, const int& z, const int& width, const int& height) {
	ind = x + width * (y + height * z);
}

/*!
 * \brief Converts 1D-array index to 3D-array index.
 * \param ind Input 1D-array index
 * \param x Output 3D-array X-index
 * \param y Output 3D-array Y-index
 * \param z Output 3D-array Z-index
 * \param width 3D-array width
 * \param height 3D-array height
 */
void hough::ind_1d_to_3d(int ind, int& x, int& y, int& z, const int& width, const int& height) {
	z = ind / (width * height);
	ind -= (z * width * height);
	y = ind / width;
	x = ind % width;
}

/*!
 * \brief Performs a circle hough transformation on an edge image with different parallelization techniques.
		  Records execution times of main hough transform algorithm.
		  Applies linear binning and euclidean spacing to filter found circles.
		  Counts all circles. Outputs image with drawn circles.
		  <A HREF=hough_8c_source.html><B> main.c annotated source </B></A>
 * \param imp_type Implementation type (sequentail, omp, mpi)
 * \param mpi_type MPI field size to send and receive
 * \param img Edge image
 * \param src_img Original colored image
 * \param min_radius Minimum circle radius
 * \param max_radius Maximum circle radius
 * \param peak_tresh Accumulator peak treshold
 * \param use_binning Binning on/off
 * \param bin_size Bin size
 * \param use_spacing Spacing on/off
 * \param spacing_size Spacing size
 * \param world_size Number of all MPI processes
 * \param world_rank Process ID of an MPI process
 * \param omp_threads Number of OpenMP threads
 */
Mat hough::circle(
	ImpType imp_type,
	MpiType mpi_type,
	Mat& img,
	Mat& src_img,
	const int& min_radius,
	const int& max_radius,
	const int& peak_tresh,
	const bool& use_binning,
	const int& bin_size,
	const bool& use_spacing,
	const int& spacing_size,
	const int& world_size,
	const int& world_rank,
	const int& omp_threads) {

#pragma region variable declaration

	vector<tuple<int, int, int, bool>> circles; //list of found circles; tuple: x,y,r,drawn
	int circles_found_cnt = 0; //number of circles found

	//accumulator-related

	int acc_w = img.cols; //accumulator width
	int acc_h = img.rows; //accumulator height
	int acc_d = (max_radius - min_radius + 1); //accumulator depth (z)
	int acc_size = acc_w * acc_h * acc_d; //accumulator total size

	ushort* acc; //accumulator 1d-array
	ushort* acc_rbuf; //accumulator receive buffer, 1d-array
	vector<ushort*> accs; //list of accumulators
	vector<tuple<int, int>> accs_sizes; //list of accumulator sizes; tuple: width,total_size
	int accs_cur_w = 0; //current width of accumulator while merging received mpi accumulators

	//image-related

	int src_x = 0; //current image X-index (also used for ROI cropping)
	int src_x2 = acc_w; //current image 2nd X-index (also used for ROI cropping)
	int src_y = 0; //image Y-index
	int src_w = img.cols; //image width
	int src_h = img.rows; //image height
	int src_size = src_w * src_h; //total image size
	uchar* src; //image 1d-array

	vector<uchar*> src_rois; //list of image ROIs
	vector<tuple<int, int, int>> src_roi_sizes; //list of image ROI sizes; tuple: x,w,size
	int src_roi_shiftsize = 0; //size of ROI shift in X-direction

	//index conversion variables

	int ind = 0; //1d-index in 2d-image
	int ind_acc = 0; //1d-index in 3d-accumulator
	int ind_x = 0, ind_y = 0, ind_z = 0; //3d-indices in 1d-array

	/*
	to_1d(ind, 1, 2, 3, acc_w, acc_h);
	cout << "to_1d: " << ind << endl;

	to_3d(ind, ind_x, ind_y, ind_z, acc_w, acc_h);
	cout << "to_3d: " << ind_x << " " << ind_y << " " << ind_z << " " << endl;
	*/

	//for mpi crop, shifting X-positions for proper accumulator coords in hough transform algorithm
	int mpi_x_shift = (imp_type == ImpType::openmpi && mpi_type == MpiType::crop) ? max_radius : 0;

	//flag to indicate whether there is enough space between each circle
	bool inbetween_ok;
	//max accumulator values found while binning
	double bin_max, bin_acc_cur;
	//hough accumulator coordinates, max coords found while binning
	int hough_x, hough_y, bin_max_r, bin_max_x, bin_max_y;
	//execution time points
	std::chrono::time_point<std::chrono::high_resolution_clock>
		time_start_total, 
		time_start_hough_nompi, 
		time_end_hough_nompi, 
		time_end_hough;

#pragma endregion

	//measuring total runtime
	time_start_total = std::chrono::high_resolution_clock::now();

#pragma region prepare image and accumulator (+ROIs)

	if (imp_type == ImpType::openmpi) {

		//mpi

		if (mpi_type == MpiType::crop && world_rank == 0) {
			//mpi crop, root, total acc matrix is width + (max_radius * 2)
			src = new uchar[src_size]();
			acc_w += (max_radius * 2);
			acc_size = acc_w * acc_h * acc_d;
			acc = new ushort[acc_size]();
		}

		//split image into <n> vertical stripes:
		src_roi_shiftsize = int(img.cols / (world_size - 1));

		for (int i = 0; i < (world_size - 1); i++) {
			int roi_x = src_roi_shiftsize * i;
			int roi_w = src_roi_shiftsize;
			if (i == world_size - 2) {
				roi_w = img.cols - roi_x;
			}

			if (mpi_type == MpiType::crop && world_rank == 0) {
				//mpi crop, root, cropping src image into multiple rois
				Mat roi = img(Rect(roi_x, 0, roi_w, src_h));
				uchar* roi_arr = new uchar[roi_w * src_h]();
				fill_img_into_2d_array(roi_arr, roi, roi_w, src_h);
				src_rois.push_back(roi_arr);
			}

			src_roi_sizes.push_back(make_tuple(roi_x, roi_w, roi_w * src_h));

			if (mpi_type == MpiType::crop) {
				//mpi crop, all processes, cropping accumulator matrices
				//roi_w + (max_radius * 2) includes external (lying outside of accumulator size) polar coordinates
				int acc_crop_w = roi_w + (max_radius * 2);
				int acc_crop_size = acc_crop_w * acc_h * acc_d;
				if (world_rank == 0) {
					//mpi crop, root, initializing empty cropped accumulator matrices
					ushort* acc_crop = new ushort[acc_crop_size]();
					accs.push_back(acc_crop);
				}
				accs_sizes.push_back(make_tuple(acc_crop_w, acc_crop_size));
			}
		}

		if (mpi_type == MpiType::full) {
			//mpi full, all processes, initializing full-sized image and accumulator matrices
			src = new uchar[src_size](); // () initializes all values to 0
			fill_img_into_2d_array(src, img, src_w, src_h);
			acc = new ushort[acc_size]();
			acc_rbuf = new ushort[acc_size]();

			if (world_rank != 0) {
				//mpi full, non-root, setting ROI X-coordinates
				src_x = get<0>(src_roi_sizes[world_rank - 1]);
				src_x2 = src_x + get<1>(src_roi_sizes[world_rank - 1]);
			}
		}

		if (mpi_type == MpiType::crop && world_rank != 0) {
			//mpi crop, non-root, initializing image and accumulator with properly cropped sizes

			//image, setting ROI limits
			src_w = get<1>(src_roi_sizes[world_rank - 1]);
			src_x2 = src_w;
			src_size = get<2>(src_roi_sizes[world_rank - 1]);
			src = new uchar[src_size]();

			//accumulator, setting sizes
			acc_w = get<0>(accs_sizes[world_rank - 1]);
			acc_size = get<1>(accs_sizes[world_rank - 1]);
			acc = new ushort[acc_size]();
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
	else {

		//implementation: seq, omp
		//initialize image, accumulator

		src = new uchar[src_size]();
		fill_img_into_2d_array(src, img, src_w, src_h); //fill 1d-array (src) with real 2d-image
		acc = new ushort[acc_size]();
	}
	

#pragma endregion

#pragma region circle hough transform

	if (imp_type == ImpType::openmpi) {

		//mpi, root, send 2d-array of image or its ROIs to every process
		if (world_rank == 0) {
			for (int i = 1; i < world_size; i++) {
				if (mpi_type == MpiType::full) {
					//mpi full, send full image
					MPI_Send(src, src_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
				}
				else {
					//testing image ROIs
					//imwrite("img" + to_string(i) + ".png", Mat(acc_h, get<1>(src_roi_sizes[i - 1]), CV_8UC1, src_rois[i - 1]));

					//mpi crop, send image ROIs
					MPI_Send(src_rois[i - 1], get<2>(src_roi_sizes[i - 1]), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
				}
			}
		}
		else {
			//recv 2d-array from root
			MPI_Recv(src, src_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		time_start_hough_nompi = std::chrono::high_resolution_clock::now(); //measuring hough runtime without mpi communication 
	}

	if (imp_type != ImpType::openmpi || (imp_type == ImpType::openmpi && world_rank != 0)) { //don't run in mpi root process

		#pragma omp parallel for num_threads(omp_threads) collapse(2) private(ind, hough_x, hough_y) shared(acc) if(imp_type == ImpType::openmp)
		//#pragma omp parallel for simd num_threads(omp_threads) collapse(2) private(ind, hough_x, hough_y) shared(acc) if(imp_type == ImpType::openmp)
		//for every image pixel
		for (int j = src_y; j < src_h; j++) {
			for (int i = src_x; i < src_x2; i++) {

				ind_2d_to_1d(ind, i, j, src_w);
				if (src[ind] == 255) { //if pixel is edge (255 = white)

				//for every radius, draw a circle (360 degrees)
					for (int r = min_radius; r <= max_radius; r++) {
						for (int t = 0; t <= 360; t++) {

							//find polar coordinates for current circle center at i,j
							//also convert degrees to radians
							hough_x = (i + mpi_x_shift) - (r * cos((t * CV_PI) / 180.0)); //mpi_x_shift for proper acc coords in mpi crop
							hough_y = j - (r * sin((t * CV_PI) / 180.0));

							//if calculated coords lie within current accumulator
							if (hough_x >= 0 && hough_x < acc_w && hough_y >= 0 && hough_y < acc_h) {

								ind_3d_to_1d(ind, hough_x, hough_y, r - min_radius, acc_w, acc_h);
								acc[ind] += 1; //vote for this accumulator position
							}
						}
					}
				}
			}
		}
	}

	//mpi, gather all accumulators from all non-root processes in root
	if (imp_type == ImpType::openmpi) {

		time_end_hough_nompi = std::chrono::high_resolution_clock::now();

		if (world_rank != 0) {
			//mpi, non-root, send accumulator to root
			MPI_Send(acc, acc_size, MPI_UNSIGNED_SHORT, 0, 0, MPI_COMM_WORLD);

		}
		else {
			//merging all retrieved accumulators into a single accumulator
			for (int i = 1; i < world_size; i++) {
				if (mpi_type == MpiType::full) {
					//mpi full, root, receive accumulators from non-root processes
					MPI_Recv(acc_rbuf, acc_size, MPI_UNSIGNED_SHORT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					//mpi full, sum all accumulator coordinates
					for (int j = 0; j < acc_size; j++) {
						if (acc_rbuf[j] != 0) {
							acc[j] += acc_rbuf[j];
						}
					}
				}
				else {
					//mpi crop, root, receive cropped accumulators from non-root processes
					MPI_Recv(accs[i - 1], get<1>(accs_sizes[i - 1]), MPI_UNSIGNED_SHORT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					//mpi crop, retrieve current cropped accumulator width (depending on process id index)
					accs_cur_w = get<0>(accs_sizes[i - 1]);

					//testing cropped accumulator images
					//imwrite("acc" + to_string(i) + ".png", Mat(acc_h, accs_cur_w, CV_16S, accs[i - 1]));

					//mpi crop, merge all coords from cropped accumulators into the total accumulator with proper X-shifts
					for (int x = 0; x < accs_cur_w; x++) {
						for (int y = 0; y < acc_h; y++) {
							for (int r = 0; r < acc_d; r++) {
								ind_3d_to_1d(ind, x, y, r, accs_cur_w, acc_h);
								ind_3d_to_1d(ind_acc, x + (src_roi_shiftsize * (i - 1)), y, r, acc_w, acc_h);
								acc[ind_acc] += accs[i - 1][ind]; //merging accumulator votes
							}
						}
					}
				}
			}

			//testing merged accumulator image
			//imwrite("acc_final.png", Mat(acc_h, acc_w, CV_16S, acc));
		}
	}

	time_end_hough = std::chrono::high_resolution_clock::now();

#pragma endregion

#pragma region binning, spacing

	if (world_rank == 0) {

		if (!use_binning) {

			//no binning

			//#pragma omp parallel for num_threads(4) collapse(3) shared(acc, circles) if(imp_type == ImpType::openmp)
			//for every bin coordinate
			for (int j = 0; j < acc_h; j += 1) {
				for (int i = mpi_x_shift; i < acc_w - mpi_x_shift; i += 1) {
					for (int r = 0; r <= max_radius - min_radius; r++) {

						ind_3d_to_1d(ind, i, j, r, acc_w, acc_h);
						if (acc[ind] >= peak_tresh) { //if bin value greater than treshold
							circles.push_back(make_tuple(i - mpi_x_shift, j, r + min_radius, !use_spacing)); //add found circle
						}
					}
				}
			}
		}
		else {

			//binning, finding local maxima per bin

			//#pragma omp parallel for num_threads(4) collapse(2) private(bin_acc_current, bin_max_val, bin_max_val_x, bin_max_val_y, bin_max_val_r) shared(acc, circles) if(imp_type == ImpType::openmp)
			//for every bin
			for (int j = 0; j < acc_h; j += bin_size) {
				for (int i = mpi_x_shift; i < acc_w - mpi_x_shift; i += bin_size) {

					bin_max = 0;
					bin_max_r = 0;
					bin_max_x = 0;
					bin_max_y = 0;

					//for every bin coordinate
					for (int y = j; y < j + min(bin_size, acc_h - j); y++) {
						for (int x = i; x < i + min(bin_size, acc_w - mpi_x_shift - i); x++) {
							for (int r = 0; r <= max_radius - min_radius; r++) {

								//finding maximum value per bin
								ind_3d_to_1d(ind, x, y, r, acc_w, acc_h);
								bin_acc_cur = acc[ind];

								if (bin_acc_cur > bin_max) {
									bin_max = bin_acc_cur;
									bin_max_x = x - mpi_x_shift;
									bin_max_y = y;
									bin_max_r = r + min_radius;
								}
							}
						}
					}

					if (bin_max >= peak_tresh) { //if maximum bin value greater than treshold
						circles.push_back(make_tuple(bin_max_x, bin_max_y, bin_max_r, !use_spacing)); //add found circle
					}
				}
			}
		}

		//spacing, euclidean distance between circles should be bigger than spacing_size
		if (use_spacing) {

			//#pragma omp parallel for num_threads(4) private(inbetween_ok) shared(circles) if(imp_type == ImpType::openmp)
			//for every found circle
			for (int i = 0; i < circles.size(); i++) {

				inbetween_ok = true;

				//compare with every other circle
				for (int j = 0; j < circles.size(); j++) {
					if (j != i) {
						//if euclidean distance is lower than spacing_size, flag circle to not be drawn
						if (get<3>(circles[j]) == true &&
							sqrt(pow(get<0>(circles[j]) - get<0>(circles[i]), 2) +
								pow(get<1>(circles[j]) - get<1>(circles[i]), 2)) <= spacing_size) {
							inbetween_ok = false;
							break;
						}
					}
				}

				//if it was properly spaced, flag circle to be drawn
				if (inbetween_ok) {
					get<3>(circles[i]) = true;
				}
			}
		}
	}

#pragma endregion

	//register execution times, calculate total runtimes

	//algorithm flow: mpi_comm_start -> hough -> mpi_comm_end -> binning -> spacing

	//total time: mpi_comm_start ---> spacing
	//hough time: mpi_comm_start ---> mpi_comm_end (mpi_comm's will be skipped for imp_type != mpi)
	//hough nompi time: hough (works only for imp_type == mpi)

	auto time_elapsed_total = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_start_total).count();
	auto time_elapsed_hough = chrono::duration_cast<chrono::nanoseconds>(time_end_hough - time_start_total).count();
	auto time_elapsed_hough_nompi = chrono::duration_cast<chrono::nanoseconds>(time_end_hough_nompi - time_start_hough_nompi).count();

	//save runtimes to list (used for average calculations in evaluation)
	globals::runtimes.push_back(make_tuple(time_elapsed_total, time_elapsed_hough, time_elapsed_hough_nompi));

	cout << world_rank << " time elapsed (total): " << (time_elapsed_total / 1000000.0) << "ms" << endl;
	cout << world_rank << " time elapsed (hough): " << (time_elapsed_hough / 1000000.0) << "ms" << endl;
	cout << world_rank << " time elapsed (hough nompi): " << (time_elapsed_hough_nompi / 1000000.0) << "ms" << endl;

	//freeing memory

	delete[] src;
	delete[] acc;
	if (imp_type == ImpType::openmpi) {
		if (mpi_type == MpiType::full) {
			delete[] acc_rbuf;
		}
		else {
			for (int i = 0; i < accs.size(); i++) {
				delete[] accs[i];
				delete[] src_rois[i];
			}
		}
	}

	//draw circles into original image, count circles

	Mat output_hough;
	src_img.copyTo(output_hough);

	for (int i = 0; i < circles.size(); i++) {
		if (get<3>(circles[i]) == true) { //circle has 'drawn' flag
			cv::circle(output_hough, Point(get<0>(circles[i]), get<1>(circles[i])), get<2>(circles[i]), Scalar(0, 0, 255), 1, LINE_4);
			std::cout << world_rank << " circle: x: " << get<0>(circles[i]) << " y: " << get<1>(circles[i]) << " r: " << get<2>(circles[i]) << '\n';
			circles_found_cnt++; //increment circle count
		}
	}

	//return circle count from image
	if (world_rank == 0) {
		cout << world_rank << " circle count: " << circles_found_cnt << endl;
	}

	//draw circle count (as text) into image
	putText(output_hough, to_string(circles_found_cnt), Point(0, 15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 1.0, LINE_AA);

	//testing output image
	//imwrite("final.png", output_hough);

	return output_hough;
}