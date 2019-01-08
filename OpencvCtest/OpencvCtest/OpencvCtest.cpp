// OpencvCtest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <time.h>


using namespace cv;
using namespace std;

int main(int argc[], char **argv)
{
	cout << "Grabing Webcam:" << argv[1] << "\n";

	clock_t clk, clklast = clock();

	VideoCapture cap;
	cap.open(atoi(argv[1]) - 1);

	if (!cap.isOpened()) {
		cerr << "Failed to get web cam: " << argv[1] << "\n";
		return -1;
	}

	Mat frame;

	while (waitKey(1) != 27) {
		clk = clock();
		cout << "Time taken: " << clk - clklast << " ms\n";
		clklast = clk;

		cap >> frame;
		imshow("gay", frame);

	}

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
