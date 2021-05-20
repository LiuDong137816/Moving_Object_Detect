#if !defined MODDYNAMIC
#define MODDYNAMIC

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>


class ModDynamic
{
public:
    ModDynamic();
    ~ModDynamic();

public:
    void OpticalFlowCorners(const cv::Mat curGrayImage, const cv::Mat precurGrayImage, const std::vector<cv::Point2f>& curPoint, 
        const std::vector<cv::Point2f>& lastPoint, std::vector<uchar>& state)const;
    
    bool judgeStable(const std::vector<cv::Point2f>& lastPoint, const std::vector<cv::Point2f>& curPoint, const std::vector<uchar>& state)const;

    void GetInitCornerPoints(const cv::Mat& curGrayImage, const cv::Mat& lastGrayImage, 
        std::vector<cv::Point2f>& curPoint, std::vector<cv::Point2f>& lastPoint, std::vector<uchar>& state)const;

    void DrawCornersInImage(cv::Mat& frame, const std::vector<cv::Point2f>& corners, const std::vector<uchar>& state, cv::Scalar scalar)const;

    void DrawFlowLine(cv::Mat& frame, const std::vector<cv::Point2f>& lastPoint, const std::vector<cv::Point2f>& curPoint, const std::vector<uchar>& state)const;

    void DrawPolarLines(const double A, const double B, const double C, cv::Mat& frame)const;

    void CalculateFundMatrix(const std::vector<cv::Point2f>& curPoint, const std::vector<cv::Point2f>& lastPoint, const std::vector<uchar>& state, cv::Mat& fundMatrix)const;

    void GetMoveCorners(const std::vector<cv::Point2f>& curPoint, const std::vector<cv::Point2f>& lastPoint, 
        const cv::Mat& fundMatrix, const std::vector<uchar>& state, std::vector<cv::Point2f>& moveCorners)const;
private:

};

ModDynamic::ModDynamic()
{
}

ModDynamic::~ModDynamic()
{
}

#endif