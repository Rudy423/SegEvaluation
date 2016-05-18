#include "stdafx.h"
#include "Evaluation.h"

Evaluation::Evaluation()
{
}


Evaluation::~Evaluation()
{
}

//Sort the index of regions in segmentation
void Evaluation::SortSegImg(Mat& segImg)
{
	int height = segImg.rows;
	int width = segImg.cols;
	vector<int> regionList;
	vector<int>::iterator iter;
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			iter = find(regionList.begin(), regionList.end(), segImg.ptr<uchar>(i)[j]);
			if (iter == regionList.end())
				regionList.push_back((segImg.ptr<uchar>(i)[j]));
		}
	}
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			for (int k = 0; k<regionList.size(); k++)
			{
				if (segImg.ptr<uchar>(i)[j] == regionList[k])
				{
					segImg.ptr<uchar>(i)[j] = k;
					break;
				}
			}
		}
	}
}

//Get points of each region
vector<vector<Point> > Evaluation::GetAreaList(Mat& segImg)
{
	int height = segImg.rows;
	int width = segImg.cols;
	int maxRegionIndex = 0;

	for (int r = 0; r<height; r++)
	{
		for (int c = 0; c<width; c++)
		{
			int tempValue = segImg.ptr<uchar>(r)[c];
			if (tempValue > maxRegionIndex)
				maxRegionIndex = tempValue;
		}
	}

	vector<vector<Point> > areaList(maxRegionIndex + 1);
	for (int r = 0; r<height; r++)
	{
		for (int c = 0; c<width; c++)
		{
			int tempValue = segImg.ptr<uchar>(r)[c];
			areaList[tempValue].push_back(Point(c, r));
		}
	}
	return areaList;
}

//Get average spectral value of each region
vector<vector<double> > Evaluation::GetMeanSpecList(Mat& srcImg, vector<vector<Point> >& areaList)
{
	vector<vector<double> > meanSpecList;
	int channel = srcImg.channels();
	for (int r = 0; r<areaList.size(); r++)
	{
		vector<double> tempMeanSpectral(channel, 0);
		for (int c = 0; c<channel; c++)
		{
			for (int p = 0; p<areaList[r].size(); p++)
			{
				Point tempPoint = areaList[r][p];
				tempMeanSpectral[c] += srcImg.at<Vec3b>(areaList[r][p])[c];
			}
			tempMeanSpectral[c] /= areaList[r].size();
		}
		meanSpecList.push_back(tempMeanSpectral);
	}
	return meanSpecList;
}

//Unsupervised evaluation method: F
double Evaluation::GetLiu(Mat& srcImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList)
{
	int channel = srcImg.channels();
	int height = srcImg.rows;
	int width = srcImg.cols;
	double F = 0;
	for (int r=0; r<areaList.size(); r++)
	{
		double tempAreaDst = 0;
		for (int p=0; p<areaList[r].size(); p++)
		{
			double tempPointDst = 0;
			for (int c = 0; c<channel; c++)
			{
				tempPointDst += pow(srcImg.at<Vec3b>(areaList[r][p])[c] - meanSpecList[r][c], 2);
			}
			tempAreaDst += tempPointDst;
		}
		F += (tempAreaDst / sqrt((double)areaList[r].size()));
	}
	F *= sqrt((double)areaList.size());
	F /= (1000 * width * height);
	return F;
}

//Unsupervised evaluation method: Q
double Evaluation::GetBorsotti(Mat& srcImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList)
{
	int channel = srcImg.channels();
	int height = srcImg.rows;
	int width = srcImg.cols;
	double Q = 0;
	for (int r = 0; r<areaList.size(); r++)
	{
		double tempAreaDst = 0;
		for (int p = 0; p<areaList[r].size(); p++)
		{
			double tempPointDst = 0;
			int i = areaList[r][p].x;
			int j = areaList[r][p].y;
			for (int c = 0; c<channel; c++)
			{
				tempPointDst += pow(srcImg.at<Vec3b>(areaList[r][p])[c] - meanSpecList[r][c], 2);
			}
			tempAreaDst += tempPointDst;
		}
		double Q1 = tempAreaDst / (1 + log10((double)areaList[r].size()));
		int N_r = 0;
		for (int r2 = 0; r2<areaList.size(); r2++)
		{
			if (areaList[r2].size() == areaList[r].size())
				N_r++;
		}
		double Q2 = (double)N_r / areaList[r].size();
		Q2 = pow(Q2,2);
		Q += (Q1 + Q2);
	}
	Q = Q*sqrt((double)areaList.size()) / (10000 * width*height);
	return Q;
}

//Unsupervised evaluation method: FRC
double Evaluation::GetRosenberger(Mat& srcImg, Mat& segImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList)
{
	int channel = srcImg.channels();
	int height = srcImg.rows;
	int width = srcImg.cols;

	vector<vector<int> > adjMat(areaList.size(), vector<int>(areaList.size(), 0));
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					if (x == 0 || y == 0)
					{
						if (i + x >= 0 && j + y >= 0 && i + x<height && j + y<width)
						{
							int srcRegion = segImg.ptr<uchar>(i)[j];
							int dstRegion = segImg.ptr<uchar>(i+x)[j+y];
							if (srcRegion != dstRegion)
							{
								adjMat[srcRegion][dstRegion]++;
								adjMat[dstRegion][srcRegion]++;
							}
						}
					}
				}
			}
		}
	}

	vector<vector<double> > specDstMat(areaList.size(), vector<double>(areaList.size(), 0));
	for (int i = 0; i<areaList.size(); i++)
	{
		for (int j = i; j<areaList.size(); j++)
		{
			if (adjMat[i][j] != 0)
			{
				double tempAreaSpecDst = 0;
				for (int c = 0; c<channel; c++)
				{
					tempAreaSpecDst += pow((meanSpecList[i][c] - meanSpecList[j][c]), 2);
				}
				tempAreaSpecDst = sqrt(tempAreaSpecDst);
				specDstMat[i][j] = tempAreaSpecDst;
				specDstMat[j][i] = tempAreaSpecDst;
			}
		}
	}

	vector<double> contrasts(areaList.size());
	for (int i = 0; i<areaList.size(); i++)
	{
		int perimeter = 0;
		for (int j1 = 0; j1<areaList.size(); j1++)
		{
			perimeter += adjMat[i][j1];
		}
		int neighbourSize = 0;
		for (int j1 = 0; j1<areaList.size(); j1++)
		{
			if (adjMat[i][j1] != 0)
				neighbourSize += areaList[j1].size();
		}
		for (int j2 = 0; j2<areaList.size(); j2++)
		{
			if (adjMat[i][j2] != 0)
			{
				double w = (double)(adjMat[i][j2]) / perimeter;
				double u = (double)(areaList[j2].size()) / neighbourSize;
				contrasts[i] += u*specDstMat[i][j2];
			}
		}
	}

	double interContrast = 0;
	for (int i = 0; i<areaList.size(); i++)
	{
		interContrast += contrasts[i];
	}
	interContrast = interContrast / areaList.size();

	double intraContrast = 0;
	for (int r = 0; r<areaList.size(); r++)
	{
		double tempAreaDst = 0;
		for (int p = 0; p<areaList[r].size(); p++)
		{
			double tempPointDst = 0;
			for (int c = 0; c<channel; c++)
			{
				int tempValue = srcImg.at<Vec3b>(areaList[r][p])[c];
				tempPointDst += pow((tempValue - meanSpecList[r][c]), 2);
			}
			tempAreaDst += sqrt(tempPointDst);
		}
		intraContrast += tempAreaDst;
	}
	intraContrast /= (width*height);
	double Frc = (interContrast - intraContrast) / 2;
	return Frc;
}

//Unsupervised evaluation method: E
double Evaluation::GetZhang(Mat& srcImg, vector<vector<Point> >& areaList)
{
	int channel = srcImg.channels();
	int height = srcImg.rows;
	int width = srcImg.cols;

	double interEntropy = 0;
	for (int i = 0; i<areaList.size(); i++)
	{
		double regionWeight = (double)(areaList[i].size()) / (width*height);
		double tempEntropy = regionWeight*log10(regionWeight);
		interEntropy += (-tempEntropy);
	}

	double intraEntropy = 0;
	for (int c = 0; c<channel; c++)
	{
		double channelEntropy = 0;
		for (int r = 0; r<areaList.size(); r++)
		{
			double regionEntropy = 0;
			vector<int> tempRegion(areaList[r].size());
			for (int p = 0; p<areaList[r].size(); p++)
				tempRegion[p] = srcImg.at<Vec3b>(areaList[r][p])[c];
			vector<double> grayProbVec(256, 0);
			for (int p = 0; p<tempRegion.size(); p++)
			{
				grayProbVec[tempRegion[p]]++;
			}
			for (int i = 0; i<grayProbVec.size(); i++)
			{
				if (grayProbVec[i] != 0)
				{
					grayProbVec[i] = grayProbVec[i] / tempRegion.size();
					double temp = grayProbVec[i] * log10(grayProbVec[i]);
					regionEntropy += (-temp);
				}
			}
			double regionWeight = (double)(areaList[r].size()) / (width*height);
			channelEntropy += regionWeight*regionEntropy;
		}
		intraEntropy += channelEntropy;
	}
	intraEntropy /= channel;

	return interEntropy + intraEntropy;
}

//Convert color space from RGB to LAB
Mat RGB2LAB(Mat& RGBImg)
{
	int channel = RGBImg.channels();
	int height = RGBImg.rows;
	int width = RGBImg.cols;
	Mat LABImg(height, width, CV_32FC3);
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			int sR = RGBImg.at<Vec3b>(Point(j, i))[0];
			int sG = RGBImg.at<Vec3b>(Point(j, i))[1];
			int sB = RGBImg.at<Vec3b>(Point(j, i))[2];
			//------------------------
			// sRGB to XYZ conversion
			// (D65 illuminant assumption)
			//------------------------
			double R = sR / 255.0;
			double G = sG / 255.0;
			double B = sB / 255.0;

			double r, g, b;

			if (R <= 0.04045)	r = R / 12.92;
			else				r = pow((R + 0.055) / 1.055, 2.4);
			if (G <= 0.04045)	g = G / 12.92;
			else				g = pow((G + 0.055) / 1.055, 2.4);
			if (B <= 0.04045)	b = B / 12.92;
			else				b = pow((B + 0.055) / 1.055, 2.4);

			double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
			double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
			double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
			//------------------------
			// XYZ to LAB conversion
			//------------------------
			double epsilon = 0.008856;	//actual CIE standard
			double kappa = 903.3;		//actual CIE standard

			double Xr = 0.950456;	//reference white
			double Yr = 1.0;		//reference white
			double Zr = 1.088754;	//reference white

			double xr = X / Xr;
			double yr = Y / Yr;
			double zr = Z / Zr;

			double fx, fy, fz;
			if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
			else				fx = (kappa*xr + 16.0) / 116.0;
			if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
			else				fy = (kappa*yr + 16.0) / 116.0;
			if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
			else				fz = (kappa*zr + 16.0) / 116.0;

			LABImg.at<Vec3f>(Point(j, i))[0] = (float)(116.0*fy - 16.0);
			LABImg.at<Vec3f>(Point(j, i))[1] = (float)(500.0*(fx - fy));
			LABImg.at<Vec3f>(Point(j, i))[2] = (float)(200.0*(fy - fz));
		}
	}
	return LABImg;
}

//Convolve LAB image with kernal [1, 4, 6, 4, 1]
Mat GaussianSmooth(Mat& LABImg)
{
	int channel = LABImg.channels();
	int height = LABImg.rows;
	int width = LABImg.cols;
	int kernel[5] = { 1, 4, 6, 4, 1 };

	//x direction
	Mat tempImg(height, width, CV_32FC3);
	for (int b = 0; b<channel; b++)
	{
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				double kernelSum = 0;
				double sum = 0;
				for (int cc = -2; cc <= 2; cc++)
				{
					if (((j + cc) >= 0) && ((j + cc) < width))
					{
						float tempPixel = LABImg.at<Vec3b>(Point(j + cc, i))[b];
						sum += tempPixel * kernel[2 + cc];
						kernelSum += kernel[2 + cc];
					}
				}
				tempImg.at<Vec3f>(Point(j, i))[b] = sum / kernelSum;
			}
		}
	}

	//y direction
	Mat smoothImg(height, width, CV_32FC3);
	for (int b = 0; b<channel; b++)
	{
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				double kernelSum = 0;
				double sum = 0;
				for (int cc = -2; cc <= 2; cc++)
				{
					if (((i + cc) >= 0) && ((i + cc) < height))
					{
						float tempPixel = tempImg.at<Vec3f>(Point(j, i+cc))[b];
						sum += tempPixel * kernel[2 + cc];
						kernelSum += kernel[2 + cc];
					}
				}
				smoothImg.at<Vec3f>(Point(j, i))[b] = sum / kernelSum;
			}
		}
	}
	return smoothImg;
}

//Get average LAB value of each region
vector<vector<double> > GetMeanLABList(Mat& LABImg, vector<vector<Point> >& areaList)
{
	vector<vector<double> > meanLABList;
	for (int r = 0; r<areaList.size(); r++)
	{
		vector<double> tempArea(3, 0);
		for (int b = 0; b<3; b++)
		{
			for (int m = 0; m<areaList[r].size(); m++)
			{
				tempArea[b] += LABImg.at<Vec3b>(areaList[r][m])[b];
			}
			tempArea[b] /= areaList[r].size();
		}
		meanLABList.push_back(tempArea);
	}
	return meanLABList;
}

//Get saliency map
vector<double> GetSaliencyVec(Mat& smoothImg, vector<vector<Point> >& areaList, vector<vector<double> >& meanLABList)
{
	int channel = smoothImg.channels();
	int height = smoothImg.rows;
	int width = smoothImg.cols;
	Mat saliencyMap(height, width, CV_32FC1);
	for (int r = 0; r<areaList.size(); r++)
	{
		for (int p = 0; p<areaList[r].size(); p++)
		{
			float tempPointSal = 0;
			for (int c = 0; c<channel; c++)
			{
				tempPointSal += pow(smoothImg.at<Vec3b>(areaList[r][p])[c] - meanLABList[r][c], 2);
			}
			saliencyMap.at<float>(areaList[r][p]) = sqrt(tempPointSal);
		}
	}
	vector<double> saliencyVec(areaList.size(), 0);
	for (int r = 0; r<areaList.size(); r++)
	{
		for (int p = 0; p<areaList[r].size(); p++)
		{
			saliencyVec[r] += saliencyMap.at<float>(areaList[r][p]);
		}
	}
	return saliencyVec;
}

//Get region adjacent graph
vector<side> GetRAG(Mat& segImg)
{
	int height = segImg.rows;
	int width = segImg.cols;

	vector<side> RAG;
	vector<side>::iterator iter;
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					if (abs(x) + abs(y) == 1)
					{
						Point tempPoint(i + x, j + y);
						if (tempPoint.x >= 0 && tempPoint.y >= 0 && tempPoint.x < height && tempPoint.y < width)
						{
							int ori = segImg.ptr<uchar>(i)[j];
							int ter = segImg.ptr<uchar>(i+x)[j+y];
							if (ori != ter)
							{
								side tempSide = { ori, ter, 0 };
								iter = find(RAG.begin(), RAG.end(), tempSide);
								if (iter == RAG.end())
									RAG.push_back(tempSide);
							}
						}
					}
				}
			}
		}
	}
	return RAG;
}

//Calculate cost of regions
void CalCost(vector<side>& RAG, vector<vector<Point> >& areaList, vector<vector<double> >& meanSpecList)
{
	int channel = meanSpecList[0].size();
	for (int i = 0; i<RAG.size(); i++)
	{
		double oriSize = areaList[RAG[i].ori].size();
		double terSize = areaList[RAG[i].ter].size();
		double p_area = (oriSize*terSize) / (oriSize+terSize);
		double p_spec = 0;
		for (int c = 0; c<channel; c++)
		{
			p_spec += pow(meanSpecList[RAG[i].ori][c] - meanSpecList[RAG[i].ter][c], 2);
		}
		p_spec /= channel;
		RAG[i].cost = p_area*p_spec;
	}
}

//Get d-intra
double Evaluation::GetIntraDst(Mat& srcImg, vector<vector<Point> >& areaList, vector<double>& saliencyVec)
{
	int regionSum = areaList.size();
	double avgSize = (double)srcImg.rows * srcImg.cols / regionSum;

	double intraDst = 0;
	for (int i = 0; i<saliencyVec.size(); i++)
	{
		double tempUniformity = 1.942*(saliencyVec[i] / avgSize);// -2.2737;
		intraDst += tempUniformity;
	}
	intraDst /= areaList.size();
	return intraDst;
}

//Get d-inter
double Evaluation::GetInterDst(Mat& srcImg, vector<side>& RAG)
{
	int regionSum = RAG.size() + 1;
	double avgSize = (double)srcImg.rows * srcImg.cols / regionSum;

	double interDst = 0;
	vector<double> disparityVec(RAG.size(), 0);
	for (int i = 0; i<RAG.size(); i++)
	{
		disparityVec[i] = sqrt(2 * RAG[i].cost / avgSize);
		interDst += disparityVec[i];
	}
	interDst /= RAG.size();
	return interDst;
}

//The proposed method
//if s=0, get absolutle segmentation quality; otherwise, get relative segmentation quality
double Evaluation::myEvaluation(Mat& srcImg, Mat& segImg, double s)
{
	vector<vector<Point> > areaList = GetAreaList(segImg);
	vector<vector<double> > meanSpecList = GetMeanSpecList(srcImg, areaList);
	Mat LABImg;
	cvtColor(srcImg, LABImg, CV_RGB2Lab);
	Mat smoothImg;
	GaussianBlur(LABImg, smoothImg, Size(5,5),0,0);

	vector<vector<double> > meanLABList = GetMeanLABList(LABImg, areaList);
	vector<double> saliencyVec = GetSaliencyVec(smoothImg, areaList, meanLABList);
	double intraDst = GetIntraDst(srcImg, areaList, saliencyVec);

	vector<side> RAG = GetRAG(segImg);
	CalCost(RAG, areaList, meanSpecList);
	double interDst = GetInterDst(srcImg, RAG);

	double s0 = sqrt((double)srcImg.rows * srcImg.cols / areaList.size());
 	if (s == 0)
		s = s0;
	double absQuality = interDst / (2 * intraDst);
	double deltaScale = (s<s0 ? s : s0) / (s>s0 ? s : s0);
	double relaQuality = absQuality*deltaScale;
	return relaQuality;
}

//Supervised evaluation method: SC
double Evaluation::GetSC(Mat& gtImg, Mat& segImg)
{
	int width = gtImg.cols;
	int height = gtImg.rows;
	vector<vector<Point> > gtAreaList = GetAreaList(gtImg);
	vector<vector<Point> > segAreaList = GetAreaList(segImg);
	vector<vector<int> > OverlapMat(gtAreaList.size(), vector<int>(segAreaList.size(), 0));
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			OverlapMat[gtImg.ptr<uchar>(i)[j]][segImg.ptr<uchar>(i)[j]]++;
		}
	}
	double SC = 0;
	for (int i = 0; i<gtAreaList.size(); i++)
	{
		double MaxCover = 0;
		for (int j = 0; j<segAreaList.size(); j++)
		{
			int IntersectSize = OverlapMat[i][j];
			int UnionSize = gtAreaList[i].size() + segAreaList[j].size() - IntersectSize;
			double CurCover = (double)IntersectSize / UnionSize;
			if (CurCover>MaxCover)
				MaxCover = CurCover;
		}
		SC += (MaxCover*gtAreaList[i].size());
	}
	SC = SC / (width*height);
	return SC;
}

//get correlation coefficient
double Evaluation::GetCorr(vector<vector<double> > SCQualityMat, vector<vector<double> > myQualityMat)
{
	int matSize = SCQualityMat.size()*SCQualityMat[0].size();
	double SCMean = 0, myMean = 0;
	for (size_t i = 0; i < SCQualityMat.size(); i++)
	{
		for (size_t j = 0; j < SCQualityMat[0].size(); j++)
		{
			SCMean += SCQualityMat[i][j];
			myMean += myQualityMat[i][j];
		}
	}
	SCMean /= matSize;
	myMean /= matSize;

	double aa=0, bb=0, ab = 0;
	for (size_t i = 0; i < SCQualityMat.size(); i++)
	{
		for (size_t j = 0; j < SCQualityMat[0].size(); j++)
		{
			double a = SCQualityMat[i][j] - SCMean;
			double b = myQualityMat[i][j] - myMean;
			aa += a*a;
			bb += b*b;
			ab += a*b;
		}
	}
	double corr = ab / sqrt(aa*bb);
	return corr;
}