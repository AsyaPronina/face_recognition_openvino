#include <alignment.hpp>

cv::Mat alignFace(const cv::Mat &srcImage, std::vector<cv::Point2f> leftEye, std::vector<cv::Point2f> rightEye)
{
    if (leftEye[1].x >= 0 && rightEye[1].x >= 0) {

        auto size = srcImage.size();
        auto width = size.width;
        auto height = size.height;
        cv::Point2f leftEyeCenter = { float((leftEye[0].x + leftEye[1].x) * 0.5), float((leftEye[0].y + leftEye[1].y) * 0.5f) };
        cv::Point2f rigthEyeCenter = { float((rightEye[1].x + rightEye[0].x) * 0.5), float((rightEye[1].y + rightEye[0].y) * 0.5f) };
        cv::Point2f eyesCenter = { (leftEyeCenter.x + rigthEyeCenter.x) * 0.5f, (leftEyeCenter.y + rigthEyeCenter.y) * 0.5f };

        double dy = (rigthEyeCenter.y - leftEyeCenter.y);
        double dx = (rigthEyeCenter.x - leftEyeCenter.x);
        double len = sqrt(dx*dx + dy*dy);
        double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

        cv::Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0f);

        cv::Mat warped;
        warpAffine(srcImage, warped, rot_mat, warped.size());

        return warped;
    }

    return cv::Mat();
}
