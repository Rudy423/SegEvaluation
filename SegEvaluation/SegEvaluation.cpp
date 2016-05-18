// SegEvaluation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "Evaluation.h"
#include <fstream>
#include <io.h>
#include <sstream>
#include <vector>
#include <direct.h>

using namespace std;
using namespace cv;
Evaluation evaluator;

int _tmain(int argc, _TCHAR* argv[])
{
	const string ROOT_PATH = "E:\\BSR\\BSDS500\\data\\"; //set the path of BSDS
	Mat srcImg = imread(ROOT_PATH + "images\\train\\140075.jpg", 1);
	Mat segImg = imread(ROOT_PATH + "gt\\train\\140075.bmp", 0);
	evaluator.SortSegImg(segImg);

	vector<vector<Point> > areaList = evaluator.GetAreaList(segImg);
	vector<vector<double> > meanSpecList = evaluator.GetMeanSpecList(srcImg, areaList);
	double F = evaluator.GetLiu(srcImg, areaList, meanSpecList);
	double Q = evaluator.GetBorsotti(srcImg, areaList, meanSpecList);
	double Frc = evaluator.GetRosenberger(srcImg, segImg, areaList, meanSpecList);
	double E = evaluator.GetZhang(srcImg, areaList);
	double QS = evaluator.myEvaluation(srcImg, segImg, 0);
	cout << "F = " << F << endl;
	cout << "Q = " << Q << endl;
	cout << "Frc = " << Frc << endl;
	cout << "E = " << E << endl;
	cout << "QS = " << Q << endl;

	system("PAUSE");
	return 0;
}
