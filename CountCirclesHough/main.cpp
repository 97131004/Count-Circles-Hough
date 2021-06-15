/*!
 *
 * \author notpavlov
 * \version 1.0
 * \copyright MIT License
 * \mainpage Documentation
 *
 * \section req_sec Requirements
 * - GCC 9.3.0
 * - OpenMP 5.0
 * - OpenMPI 4.0.4
 * - OpenCV 4.1.0
 * 
 * \section compile_sec Compilation
 * Compile the program using <i>make</i> build system:<br>
 * \code{.sh}
 * make
 * make clean
 * \endcode
 *
 * \section execution_sec Execution
 * Execute the progam using:<br>
 * \code{.sh}
 * [mpiexec -n <number of mpi processes>] ./CountCirclesHough [<parameters>]
 * \endcode
 * Example:
 * \code{.sh}
 * ./CountCirclesHough images/money2.png -imp=1 -omp-threads=4 -mpi=0 -gui=1 -eval-times=10 -blur=0 -blur-ksize=5 -edges=1 -edges-ksize=3 -sobel-bw-tresh=128 -canny-tresh1=275 -canny-tresh2=125 -min-radius=25 -max-radius=35 -peak-tresh=135 -use-binning=1 -bin-size=40 -use-spacing=1 -spacing-size=40
 * \endcode
 * 
 * \section gui_sec GUI
 * Press the <b>R</b> key in a GUI window to rerun all algorithms and redraw output images.
 * 
 */

#include "globals.h"
#include "blur.h"
#include "edges.h"
#include "hough.h"
#include <thread>

using namespace cv;
using namespace std;

const char* win_blur = "blur"; //!< Blur window name (GUI).
const char* win_edges = "edge detection"; //!< Edge detection window name (GUI).
const char* win_hough = "hough"; //!< Hough transform window name (GUI).

//output images

/*! \brief Input color image. */
Mat input_color;
/*! \brief Input image converted to grayscale. */
Mat input_gs;
/*! \brief Output blurred image. */
Mat output_blur;
/*! \brief Output image with found edges. */
Mat output_edges;
/*! \brief Output input image with drawn circles and circle count. */
Mat output_hough;

//input parameters

/*! \brief Currently active implementation type. */
ImpType imp_type = ImpType::openmp;
/*! \brief Currently active MPI field size to send and receive. */
MpiType mpi_type = MpiType::full;
/*! \brief Currently active blur filter. */
BlurType blur_type = BlurType::median;
/*! \brief Currently active edge detection algorithm. */
EdgesType edges_type = EdgesType::canny;

bool gui = true; //!< GUI on/off (if false, runs evaluation).
int eval_times = 10; //!< Number of times to run evaluation on hough.
int omp_threads = 4; //!< Number of OpenMP threads.
int blur_ksize = 5; //!< Blur kernel size (must be odd, between 1 to 21).
int edges_ksize = 3; //!< Edges kernel size (must be 3, 5 or 7).
int sobel_bw_tresh = 128; //!< Sobel filter, treshold for black/white (binary) image generation (0-255).
int canny_tresh1 = 275; //!< Canny filter, 1st threshold for hysteresis (0-500).
int canny_tresh2 = 125; //!< Canny filter, 2nd threshold for hysteresis (0-500).
int min_radius = 25; //!< Minimum circle radius (1-200).
int max_radius = 35; //!< Maximum circle radius (1-200).
int peak_tresh = 135; //!< Accumulator peak treshold (0-500).
bool use_binning = true; //!< Binning on/off.
int bin_size = 30; //!< Bin size (5-200).
bool use_spacing = true; //!< Spacing on/off.
int spacing_size = 40; //!< Spacing size (0-200).

//mpi-related fields

int world_size = 0; //!< Number of all MPI processes.
int world_rank = 0; //!< Process ID of an MPI process.

/*!
* \brief Struct of all parameters to be updated for each MPI process.
         Will be sent from root and received at other MPI processes.
		 Parameters are the same as in the \link main.cpp \endlink file.
*/
struct params_update {
	int min_radius;
	int max_radius;
	int peak_tresh;
	int bin_size;
	int spacing_size;
};

/*!
* \brief Fixes value limits for all input parameters.
*/
void fix_vals() {
	if (bin_size < 5) bin_size = 5;
	//cout << world_rank << " bin_size: " << to_string(bin_size) << endl;

	if (min_radius < 1) min_radius = 1;
	max_radius = max(min_radius, max_radius);
	//cout << world_rank << " radius: " << to_string(min_radius) << " -> " << to_string(max_radius) << endl;

	blur_ksize = max(1, min(21, blur_ksize));
	blur_ksize = (blur_ksize % 2 == 0) ? blur_ksize + 1 : blur_ksize;
	//cout << world_rank << " blur ksize: " << to_string(blur_ksize) << endl;

	edges_ksize = max(3, min(7, edges_ksize));
	edges_ksize = (edges_ksize % 2 == 0) ? edges_ksize + 1 : edges_ksize;
	//cout << world_rank << " edges ksize: " << to_string(edges_ksize) << endl;

	if (gui) {
		setTrackbarPos("min radius", win_hough, min_radius);
		setTrackbarPos("ksize", win_blur, blur_ksize);
		setTrackbarPos("ksize", win_edges, edges_ksize);
		setTrackbarPos("max radius", win_hough, max_radius);
		setTrackbarPos("bin size", win_hough, bin_size);
	}
}

/*!
* \brief Runs hough transform to find and count all circles. Outputs image with found circles.
*/
void do_hough() {

	cout << "\n" << world_rank << " loading.." << endl;

	output_hough = hough::circle(
		imp_type,
		mpi_type,
		output_edges,
		input_color,
		min_radius,
		max_radius,
		peak_tresh,
		use_binning,
		bin_size,
		use_spacing,
		spacing_size,
		world_size,
		world_rank,
		omp_threads);

	cout << world_rank << " done.\n" << endl;

	if (world_rank == 0) {
		imshow(win_hough, output_hough);
	}
}

/*!
* \brief Runs active edge detection algorithm, calls hough transform. Outputs edge image.
*/
void do_edges() {

	if (edges_type == EdgesType::canny) {
		output_edges = edges::canny(output_blur, canny_tresh1, canny_tresh2, edges_ksize);
	}
	else if (edges_type == EdgesType::sobel) {
		output_edges = edges::sobel(output_blur, sobel_bw_tresh, edges_ksize);
	}

	if (world_rank == 0) {
		imshow(win_edges, output_edges);
	}
	do_hough();
}

/*!
* \brief Runs active blur filter, calls edge detection. Outputs blur image.
*/
void do_blur() {

	if (blur_type == BlurType::median) {
		output_blur = blur::median(input_gs, blur_ksize);
	}
	else if (blur_type == BlurType::gaussian) {
		output_blur = blur::gaussian(input_gs, blur_ksize);
	}

	if (world_rank == 0) {
		imshow(win_blur, output_blur);
	}
	do_edges();
}

/*!
 * \brief Main program method. 
 *        Updates GUI, runs blur, edge detection and hough transform algorithms. 
 *        Visualizes output images.
 * \param argc Input arguments count
 * \param argv Array of input arguments
 */
int main(int argc, char* argv[])
{
	Mat src; //input image

	//reading input parameters with OpenCV's CommandLineParser

	//src = imread(argv[1], IMREAD_COLOR);
	//src = imread("../images/money2.png", IMREAD_COLOR);
	
	const cv::String keys =
		"{@img||}"
		"{imp|0|}"
		"{mpi|0|}"
		"{edges|0|}"
		"{blur|0|}"
		"{eval-times|10|}"
		"{omp-threads|2|}"
		"{gui|1|}"
		"{blur-ksize|5|}"
		"{edges-ksize|3|}"
		"{sobel-bw-tresh|128|}"
		"{canny-tresh1|100|}"
		"{canny-tresh2|200|}"
		"{min-radius|15|}"
		"{max-radius|30|}"
		"{peak-tresh|125|}"
		"{use-binning|1|}"
		"{bin-size|32|}"
		"{use-spacing|1|}"
		"{spacing-size|40|}";

	cv::CommandLineParser cmd(argc, argv, keys);

	imp_type = static_cast<ImpType>(cmd.get<int>("imp"));
	mpi_type = static_cast<MpiType>(cmd.get<int>("mpi"));
	edges_type = static_cast<EdgesType>(cmd.get<int>("edges"));
	blur_type = static_cast<BlurType>(cmd.get<int>("blur"));
	eval_times = cmd.get<int>("eval-times");
	omp_threads = cmd.get<int>("omp-threads");
	gui = cmd.get<int>("gui");

	blur_ksize = cmd.get<int>("blur-ksize");
	sobel_bw_tresh = cmd.get<int>("sobel-bw-tresh");
	canny_tresh1 = cmd.get<int>("canny-tresh1");
	canny_tresh2 = cmd.get<int>("canny-tresh2");
	edges_ksize = cmd.get<int>("edges-ksize");
	min_radius = cmd.get<int>("min-radius");
	max_radius = cmd.get<int>("max-radius");
	peak_tresh = cmd.get<int>("peak-tresh");
	use_binning = cmd.get<int>("use-binning");
	bin_size = cmd.get<int>("bin-size");
	use_spacing = cmd.get<int>("use-spacing");
	spacing_size = cmd.get<int>("spacing-size");

	src = imread(cmd.get<string>("@img"), IMREAD_COLOR);

	if (!src.data)
	{
		cout << "Input data invalid." << endl;
		getchar(); //
		return -1;
	}

	//convert rgb to grayscale image

	src.copyTo(input_color);

	cv::cvtColor(src, src, COLOR_BGR2GRAY);
	/*
	cv::imwrite("../images/bw.png", src);
	*/
	src.copyTo(input_gs);


	//init mpi

	struct params_update params;
	MPI_Datatype params_update;

	if (imp_type == ImpType::openmpi) {
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		//cout << "world_size: " << world_size << " world_rank:" << world_rank << endl;

		//declaring new 'parameters update' data type to send it with mpi
		MPI_Datatype type[5] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
		int blocklen[5] = { 1, 1, 1, 1, 1 };
		MPI_Aint disp[5] = { sizeof(int) * 0, sizeof(int) * 1, sizeof(int) * 2, sizeof(int) * 3, sizeof(int) * 4 };
		MPI_Type_create_struct(5, blocklen, disp, type, &params_update);
		MPI_Type_commit(&params_update);
	}

	//drawing windows and trackbars

	if (gui) {

		//gui mode

		if (world_rank == 0) {
			namedWindow(win_blur);
			namedWindow(win_edges);
			namedWindow(win_hough, WINDOW_FULLSCREEN);
		}

		createTrackbar("ksize", win_blur, &blur_ksize, 21);

		if (edges_type == EdgesType::canny) {
			createTrackbar("tresh1", win_edges, &canny_tresh1, 500);
			createTrackbar("tresh2", win_edges, &canny_tresh2, 500);
		}
		else if (edges_type == EdgesType::sobel) {
			createTrackbar("bw tresh", win_edges, &sobel_bw_tresh, 255);
		}
		createTrackbar("ksize", win_edges, &edges_ksize, 7);

		createTrackbar("min radius", win_hough, &min_radius, 200);
		createTrackbar("max radius", win_hough, &max_radius, 200);
		createTrackbar("peak tresh", win_hough, &peak_tresh, 500);

		if (use_binning) {
			createTrackbar("bin size", win_hough, &bin_size, 200);
		}

		if (use_spacing) {
			createTrackbar("spacing", win_hough, &spacing_size, 200);
		}

		fix_vals(); //fix parameter value limits

		do_blur(); //runs blur, edge detection, hough transform

		//intercepting 'R' key button to generate images with new settings

		while (true) {

			if (world_rank == 0) {
				char key = (char)waitKey(0);

				if (key == 'r' || key == 'R') //'R' pressed
				{
					//refresh all parameters and redo hough

					fix_vals();

					if (imp_type == ImpType::openmpi) {

						//fill parameters update struct to be send
						params.bin_size = bin_size;
						params.max_radius = max_radius;
						params.min_radius = min_radius;
						params.peak_tresh = peak_tresh;
						params.spacing_size = spacing_size;

						//send updated parameters to all mpi processes
						for (int i = 1; i < world_size; i++) {
							MPI_Send(&params, 1, params_update, i, 0, MPI_COMM_WORLD);
						}
					}

					do_blur();
				}
			}
			else {

				//receive updated parameters from root process
				MPI_Recv(&params, 1, params_update, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				//cout << world_rank << " upd: " << params.spacing << " " << params.max_radius << endl;

				//refresh all parameters and redo hough

				bin_size = params.bin_size;
				max_radius = params.max_radius;
				min_radius = params.min_radius;
				peak_tresh = params.peak_tresh;
				spacing_size = params.spacing_size;

				fix_vals();
				do_blur();
			}
		}
	}
	else {

		//evaluation mode

		fix_vals();

		if (blur_type == BlurType::median) {
			output_blur = blur::median(input_gs, blur_ksize);
		}
		else if (blur_type == BlurType::gaussian) {
			output_blur = blur::gaussian(input_gs, blur_ksize);
		}

		if (edges_type == EdgesType::canny) {
			output_edges = edges::canny(output_blur, canny_tresh1, canny_tresh2, edges_ksize);
		}
		else if (edges_type == EdgesType::sobel) {
			output_edges = edges::sobel(output_blur, sobel_bw_tresh, edges_ksize);
		}

		//record execution times of hough

		long long sum_total = 0;
		long long sum_hough = 0;
		long long sum_mpi = 0;

		for (int i = 0; i < eval_times; i++) {

			if (imp_type == ImpType::openmpi) {
				MPI_Barrier(MPI_COMM_WORLD);
			}

			output_hough = hough::circle(
				imp_type,
				mpi_type,
				output_edges,
				input_color,
				min_radius,
				max_radius,
				peak_tresh,
				use_binning,
				bin_size,
				use_spacing,
				spacing_size,
				world_size,
				world_rank,
				omp_threads);

			cout << endl;

			sum_total += get<0>(globals::runtimes[i]);
			sum_hough += get<1>(globals::runtimes[i]);
			sum_mpi += get<2>(globals::runtimes[i]);
		}

		//calculating average

		double avg_total = sum_total / (eval_times * 1000000.0);
		double avg_hough = sum_hough / (eval_times * 1000000.0);
		double avg_hough_nompi = sum_mpi / (eval_times * 1000000.0);

		cout << world_rank << " time elapsed avg (total): " << avg_total << " ms" << endl;
		cout << world_rank << " time elapsed avg (hough): " << avg_hough << " ms" << endl;
		cout << world_rank << " time elapsed avg (hough nompi): " << avg_hough_nompi << " ms" << endl;

		//write average to file

		std::ofstream times;
		times.open("avg.txt", std::ios_base::app);
		times << world_rank << ";" << imp_type << ";" << avg_total << ";" << avg_hough << ";" << avg_hough_nompi << std::endl;
	}

	//finalize mpi

	if (imp_type == ImpType::openmpi) {
		MPI_Finalize();
	}

	//getchar();

	return 0;
}
