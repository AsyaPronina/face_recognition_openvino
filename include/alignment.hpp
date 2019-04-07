#pragma once

#include <opencv2/opencv.hpp>

const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80; // Controls how tall the face mask is.

const int DESIRED_FACE_WIDTH = 70;
const int DESIRED_FACE_HEIGHT = DESIRED_FACE_WIDTH;

cv::Mat alignFace(const cv::Mat &srcImage, std::vector<cv::Point2f> leftEye, std::vector<cv::Point2f> rightEye);
