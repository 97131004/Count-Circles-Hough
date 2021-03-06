Parallel Systems - SoSe 2020 - Group 02

# Count Circles Hough

Tool to count the number of circles in a 2D image (view from above) using Circle Hough Transform (CHT) + visualization. For this, an edge detection algorithm is first applied on the source image. Binning and spacing algorithms are then used to eliminate overlapping circles.

![screenshot](https://github.com/97131004/Count-Circles-Hough/blob/main/screenshot.jpg?raw=true)

## Requirements

* GCC 9.3.0
* OpenMP 5.0
* OpenMPI 4.0.4
* OpenCV 4.1.0

## Installation

Compile the program using the *make* build system:
```
make
make clean
```

## Execution

Execute the progam using the command line:

```
[mpiexec -n <number of mpi processes>] ./CountCirclesHough [<parameters>]
```

Example:

```
./CountCirclesHough images/money2.png -imp=1 -omp-threads=4 -mpi=0 -gui=1 -eval-times=10 -blur=0 -blur-ksize=5 -edges=1 -edges-ksize=3 -sobel-bw-tresh=128 -canny-tresh1=275 -canny-tresh2=125 -min-radius=25 -max-radius=35 -peak-tresh=135 -use-binning=1 -bin-size=40 -use-spacing=1 -spacing-size=40
```

## GUI

Press the **R** key in a GUI window to rerun all algorithms and redraw all output images.

## License

MIT

Money pictures at **CountCirclesHough/images** are taken from *[F. S. TASEL und A. TEMIZEL, „Parallelization of Hough Transform for Circles using CUDA,“ [Online] [Accessed 13 July, 2020]](http://developer.download.nvidia.com/GTC/PDF/GTC2012/Posters/P0438_ht_poster_gtc2012.pdf)*.