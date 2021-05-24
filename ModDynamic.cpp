#include <iostream>
#include <string>
#include<io.h>
#include "ModDynamic.h"

using namespace cv;
using namespace std;

///调参点
double scale = 1; //设置缩放倍数
int margin = 4; //帧间隔
double limit_dis_epi = 2; //距离极线的距离

ModDynamic::ModDynamic()
{
    scale = 0.5;
}

void ModDynamic::setFrameROI(Rect rect)
{
    ROI = rect;
}

bool ModDynamic::ROI_mod(Size frameSize, int x1, int y1)const
{
	if (x1 >= frameSize.width / 16 && x1 <= 15 * frameSize.width / 16 && y1 >= frameSize.height / 3 && y1 <= 5 * frameSize.height / 6)
        return true;
	return false;
}

void ModDynamic::OpticalFlowCorners(const Mat& curGrayImage, const Mat precurGrayImage, const vector<Point2f>& curPoint, const vector<Point2f>& lastPoint, vector<uchar>& state)const
{
	const int limit_edge_corner = 5;
    double limit_of_check = 2120;
    int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	for (int i = 0; i < state.size(); i++)
		if (state[i] != 0)
		{
		   int x1 = lastPoint[i].x, y1 = lastPoint[i].y;
		   int x2 = curPoint[i].x, y2 = curPoint[i].y;
		   if ((x1 < limit_edge_corner || x1 >= curGrayImage.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= curGrayImage.cols - limit_edge_corner
			|| y1 < limit_edge_corner || y1 >= curGrayImage.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= curGrayImage.rows - limit_edge_corner))
		   {
			   state[i] = 0;
			   continue;
		   }
		    double sum_check = 0;
		    for (int j = 0; j < 9; j++)
			    sum_check += abs(precurGrayImage.at<uchar>(y1 + dy[j], x1 + dx[j]) - curGrayImage.at<uchar>(y2 + dy[j], x2 + dx[j]));
		    if (sum_check>limit_of_check)
                state[i] = 0;
		}
	return;
}

bool ModDynamic::judgeStable(const vector<Point2f>& lastPoint, const vector<Point2f>& curPoint, const vector<uchar>& state)const
{
    int Harris_num = 0;
	int stable_num = 0;
	double limit_stalbe = 0.5;
	for (int i = 0; i < state.size(); i++)
		if (state[i])
		{
		    if (sqrt((lastPoint[i].x - curPoint[i].x)*(lastPoint[i].x - curPoint[i].x) + (lastPoint[i].y - curPoint[i].y)*(lastPoint[i].y - curPoint[i].y)) < limit_stalbe) 
                stable_num++;
		}
	if (stable_num*1.0 / Harris_num > 0.2)
        return true;
	return false;
}

#if 0
void ModDynamic::GetInitCornerPoints(const Mat& curGrayImage, const Mat& lastGrayImage, vector<Point2f>& curPoint, vector<Point2f>& lastPoint, vector<uchar>& state)const
{
    double t = (double)cvGetTickCount();
    vector<float> err;
    goodFeaturesToTrack(lastGrayImage, lastPoint, 200, 0.01, 8, Mat(), 3, false, 0.04);
    cout << "cost timeA: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    cornerSubPix(lastGrayImage, lastPoint, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
    cout << "cost timeB: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    calcOpticalFlowPyrLK(lastGrayImage, curGrayImage, lastPoint, curPoint, state, err, Size(22, 22), 5, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
    cout << "cost timeC: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
}
#else
void ModDynamic::GetInitCornerPoints(const Mat& curGrayImage, const Mat& lastGrayImage, vector<Point2f>& curPoint, vector<Point2f>& lastPoint, vector<uchar>& state)const
{
    double t = (double)cvGetTickCount();
    vector<float> err;
    Mat mask = Mat::zeros(curGrayImage.size(), CV_8UC1);
    mask(ROI).setTo(255);
    goodFeaturesToTrack(curGrayImage, curPoint, 100, 0.01, 8, mask, 3, true, 0.04);
    cout << "cost timeA: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    cornerSubPix(curGrayImage, curPoint, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
    cout << "cost timeB: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    calcOpticalFlowPyrLK(curGrayImage, lastGrayImage, curPoint, lastPoint, state, err, Size(22, 22), 5, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
    cout << "cost timeC: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
}
#endif

void ModDynamic::DrawCornersInImage(Mat& frame, const vector<Point2f>& corners, const vector<uchar>& state, Scalar scalar)const
{
    for (int i = 0; i < state.size(); i++)
    {
        if (state[i] != 0)
        {
            cv::circle(frame, corners[i], 3, scalar);
        }
    }
}

void ModDynamic::DrawFlowLine(Mat& frame, const vector<Point2f>& lastPoint, const vector<Point2f>& curPoint, const vector<uchar>& state)const
{
    for (int i = 0; i < state.size(); i++)
    {
        if (state[i] != 0)
        {
            int x1 = (int)lastPoint[i].x, y1 = (int)lastPoint[i].y;
            int x2 = (int)curPoint[i].x, y2 = (int)curPoint[i].y;
            line(frame, Point((int)lastPoint[i].x, (int)lastPoint[i].y), Point((int)curPoint[i].x, (int)curPoint[i].y), Scalar(255, 255, 0), 2);
        }
    }
}

void ModDynamic::DrawPolarLines(const double A, const double B, const double C, Mat& frame)const
{
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    if (fabs(B) < 0.0001)
    {
        x1 = C / A, y1 = 0;
        x2 = C / A, y2 = frame.cols;
    }
    else
    {
        x1 = 0, y1 = -C / B;
        x2 = frame.cols, y2 = -(C + A*frame.cols) / B;
    }
   
    line(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 0), 1);
}

void ModDynamic::CalculateFundMatrix(const vector<Point2f>& curPoint, const vector<Point2f>& lastPoint, const vector<uchar>& state, Mat& fundMatrix)const
{
    Mat mask;
    double sumDistant = 110;
    vector<Point2f> initLastPoint, initCurPoint;
    size_t stateSize = state.size();
    for(size_t i = 0; i < stateSize; ++i)
    {
        if (state[i])
        {
            initLastPoint.push_back(lastPoint[i]);
            initCurPoint.push_back(curPoint[i]);
        }
    }
    
    while (sumDistant > 5)
    {
        sumDistant = 0;
        vector<Point2f> optLastPoint, optCurPoint;
        if(initLastPoint.size() == 0 || initCurPoint.size() == 0)
        {
            sumDistant = 0;
        }
        fundMatrix = findFundamentalMat(initLastPoint, initCurPoint, mask, FM_RANSAC, 0.1, 0.99);
        for (int i = 0; i < mask.rows; i++)
        {
            if (mask.at<uchar>(i, 0) != 0)
            {
                double A = fundMatrix.at<double>(0, 0)*initLastPoint[i].x + fundMatrix.at<double>(0, 1)*initLastPoint[i].y + fundMatrix.at<double>(0, 2);
                double B = fundMatrix.at<double>(1, 0)*initLastPoint[i].x + fundMatrix.at<double>(1, 1)*initLastPoint[i].y + fundMatrix.at<double>(1, 2);
                double C = fundMatrix.at<double>(2, 0)*initLastPoint[i].x + fundMatrix.at<double>(2, 1)*initLastPoint[i].y + fundMatrix.at<double>(2, 2);
                double distant = fabs(A*initCurPoint[i].x + B*initCurPoint[i].y + C) / sqrt(A*A + B*B);
                sumDistant += distant;
                if(distant <= 0.1)
                {
                    optLastPoint.push_back(initLastPoint[i]);
                    optCurPoint.push_back(initCurPoint[i]);
                }
            }
        }
        initLastPoint = optLastPoint;
        initCurPoint = optCurPoint;
        cout << "-----sumDistant--------- " << sumDistant << "      ---------------" << endl;
    } 
}

void ModDynamic::GetMoveCorners(const vector<Point2f>& curPoint, const vector<Point2f>& lastPoint, const Mat& fundMatrix, const vector<uchar>& state, vector<Point2f>& moveCorners)const
{
    size_t pointSize = curPoint.size();
    for (size_t i = 0; i < pointSize; ++i)
    {
        if (state[i] != 0)
        {
            double A = fundMatrix.at<double>(0, 0)*lastPoint[i].x + fundMatrix.at<double>(0, 1)*lastPoint[i].y + fundMatrix.at<double>(0, 2);
            double B = fundMatrix.at<double>(1, 0)*lastPoint[i].x + fundMatrix.at<double>(1, 1)*lastPoint[i].y + fundMatrix.at<double>(1, 2);
            double C = fundMatrix.at<double>(2, 0)*lastPoint[i].x + fundMatrix.at<double>(2, 1)*lastPoint[i].y + fundMatrix.at<double>(2, 2);
            double distant = fabs(A*curPoint[i].x + B*curPoint[i].y + C) / sqrt(A*A + B*B);

            if (distant <= limit_dis_epi)
                continue;
            moveCorners.push_back(curPoint[i]);
        }
    }
}

void ModDynamic::GetGridAreaCorners(const Size& frameSize, const vector<Point2f>& moveCorners, int gridWidth, double gridCornersNum[100][100])const
{
    for (int i = 0; i < frameSize.height / gridWidth; i++)
        for (int j = 0; j < frameSize.width / gridWidth; j++)
        {
            double x1 = i*gridWidth + gridWidth / 2;
            double y1 = j*gridWidth + gridWidth / 2;
            for (int k = 0; k < moveCorners.size(); k++)
                if (ROI_mod(frameSize, moveCorners[k].x, moveCorners[k].y) && 
                    sqrt((moveCorners[k].x - y1)*(moveCorners[k].x - y1) + (moveCorners[k].y - x1)*(moveCorners[k].y - x1)) < gridWidth*sqrt(2)) 
                    gridCornersNum[i][j]++;
        }
}

void ModDynamic::DrawMovingObject(Mat& frame, const Size& frameSize, int gridWidth, double gridCornersNum[100][100])const
{
    rectangle(frame, Point(frame.cols / 16, frame.rows * 5 / 6 ), Point(15 * frame.cols / 16, frame.rows / 3), Scalar(255, 0, 0), 1);
    int rec_width = frame.cols / 25;
    double mm = 0;
    int mark_i = 0, mark_j = 0;
    for (int i = 0; i < frameSize.height / gridWidth; i++)
        for (int j = 0; j < frameSize.width / gridWidth; j++)
            if (ROI_mod(frameSize, j*gridWidth, i*gridWidth) && gridCornersNum[i][j] > mm)
            {
                mark_i = i;
                mark_j = j;
                mm = gridCornersNum[i][j];
                if (mm < 2) 
                    continue;
                rectangle(frame, Point(mark_j*gridWidth / scale - rec_width, mark_i*gridWidth / scale + rec_width), 
                    Point(mark_j*gridWidth / scale + rec_width, mark_i*gridWidth / scale - rec_width), Scalar(0, 255, 255), 3);
            }
}

void ModDynamic::MovingObjectDetect()
{
    double t = (double)cvGetTickCount();
    Mat curGrayImage;
   
    if(1.0 != scale)
    {
        resize(frame, frame, Size(frame.cols*scale, frame.rows*scale));
    }
    cvtColor(frame, curGrayImage, CV_BGR2GRAY);
    if(lastGrayFrame.empty())
    {
        std::swap(lastGrayFrame, curGrayImage);
        setFrameROI(Rect(frame.cols / 16, frame.rows / 3, 7 * frame.cols / 8, frame.rows / 2));
        return;
    }
    
    Mat fundMatrix;
    vector<Point2f> lastPoint, curPoint;
    vector<uchar> state;
    GetInitCornerPoints(curGrayImage, lastGrayFrame, curPoint, lastPoint, state);
    cout << "cost time1: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;

    OpticalFlowCorners(curGrayImage, lastGrayFrame, curPoint, lastPoint, state);

    //mod.DrawCornersInImage(curFrame, curPoint, state, Scalar(0, 0, 255));
    //mod.DrawCornersInImage(lastFrame, lastPoint, state, Scalar(0, 0, 255));
    cout << "cost timeD: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    CalculateFundMatrix(curPoint, lastPoint, state, fundMatrix);
    cout << "cost timeE: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    //T：异常角点集
    vector<Point2f> moveObjectCorners;
    GetMoveCorners(curPoint, lastPoint, fundMatrix, state, moveObjectCorners);

    int gridWidth = 10;
    double gridCornersNum[100][100] = {0};
    GetGridAreaCorners(curGrayImage.size(), moveObjectCorners, gridWidth, gridCornersNum);
    DrawMovingObject(frame, frame.size(), gridWidth, gridCornersNum);
    imshow("frame", frame);
    std::swap(curGrayImage, lastGrayFrame);
    //imshow("frame", frame);
}

void printRunTime(string str, double beginTime)
{
    cout << str << "： " << 
        ((double)cvGetTickCount() - beginTime) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
}

string findVideoFile()
{
    string filePlayName;
    vector<string> fileNames;
    std::string inPath = "*.avi";
    struct _finddata_t fileinfo;
    long handle = _findfirst(inPath.c_str(),&fileinfo);
    if(handle == -1)
        return filePlayName;
    do
    {
        fileNames.push_back(fileinfo.name);
    } while (!_findnext(handle,&fileinfo));
    _findclose(handle);
    
    if(fileNames.size() > 1)
    {
        int num;
        cout << "-----------Please choose the video.--------------" << endl;
        for(vector<string>::iterator it = fileNames.begin(); it != fileNames.end(); ++it)
        {
            char showData[100] = {0};
            sprintf_s(showData, "%d. %s.", it - fileNames.begin(), it->c_str());
            cout << showData << endl;
        }
        num = 3;
        /*while (cin >> num)
        {
            if(num < fileNames.size())
            {
                break;
            }
            else
            {
                cout << "File is not exit." << endl;
            }
        }*/
        filePlayName = fileNames[num];
    }
    else
    {
        filePlayName = fileNames[0];
    }
    return filePlayName;
}

int main(int, char**)
{
    int frameCount = 0;
	VideoCapture cap;
	cap.open(findVideoFile());
	if (!cap.isOpened())
		return -1;

    Mat frame;
    ModDynamic mod;
    while (true)
	{
		double t = (double)cvGetTickCount();
		cap >> frame;
		if (frame.empty())
            break;
        mod.setFrame(frame);

		frameCount++;
		if (frameCount % margin != 0)
		{
			continue;
		}
        
		mod.MovingObjectDetect();
        cout << "cost timeF: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
        double delay = 1000./cap.get(CV_CAP_PROP_FPS) - ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.);
        delay = delay > 0? delay : 27;

        if (waitKey(delay) >= 0)
            break;
		cout << "cost timeG: " << ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
	}
	return 0;
}

