output: main.o blur.o edges.o hough.o globals.o
	mpic++ -g main.o blur.o edges.o hough.o globals.o -o CountCirclesHough `pkg-config --cflags --libs opencv` -fopenmp

main.o: main.cpp
	mpic++ -g -c main.cpp

blur.o: blur.cpp blur.h
	mpic++ -g -c blur.cpp

edges.o: edges.cpp edges.h
	mpic++ -g -c edges.cpp

hough.o: hough.cpp hough.h
	mpic++ -g -c hough.cpp

globals.o: globals.cpp globals.h
	mpic++ -g -c globals.cpp

clean:
	rm *.o CountCirclesHough
