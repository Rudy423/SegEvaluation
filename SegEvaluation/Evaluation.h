#pragma once
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;
using namespace cv;

struct side
{
	int ori;
	int ter;
	double cost;
	bool operator==(const side& s) const
	{
		if ((s.ori == ori && s.ter == ter) || (s.ori == ter && s.ter == ori))
			return true;
		else
			return false;
	}
};

class Evaluation
{
public:
	Evaluation();
	~Evaluation();

	void SortSegImg(Mat& segImg);
	vector<vector<Point> > GetAreaList(Mat& segImg);
	vector<vector<double> > GetMeanSpecList(Mat& srcImg, vector<vector<Point> >& areaList);

	double GetLiu(Mat& srcImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList);
	double GetBorsotti(Mat& srcImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList);
	double GetRosenberger(Mat& srcImg, Mat& segImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList);
	double GetZhang(Mat& srcImg, vector<vector<Point> >& areaList);

	double GetIntraDst(Mat& srcImg, vector<vector<Point> >& areaList, vector<double>& saliencyVec);
	double GetInterDst(Mat& srcImg, vector<side>& MST);
	double myEvaluation(Mat& srcImg, Mat& segImg, double s);
	double GetSC(Mat& gtImg, Mat& segImg);
	double GetCorr(vector<vector<double> > SCQualityMat, vector<vector<double> > myQualityMat);
};
