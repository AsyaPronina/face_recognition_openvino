// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <cstdlib>

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#include <ext_list.hpp>

#include "utility.hpp"
#include "detectors.hpp"
#include "alignment.hpp"
#include "feature_extractor.hpp"
#include "classifier.hpp"

std::string faceDetectionModel = "/home/asyadev/Study/DL/face_recognition_openvino_prototype/models/face-detection-adas-0001.xml";
std::string facialLandmarksModel = "/home/asyadev/Study/DL/face_recognition_openvino_prototype/models/facial-landmarks-35-adas-0001.xml";
std::string featureExtractionModel = "/home/asyadev/Study/DL/face_recognition_openvino_prototype/models/Sphereface.xml";

using namespace InferenceEngine;

std::string retrievePath(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    if (argc == 2)
    {
        return argv[1];
    }

    return "";
}

bool recognizeFace(unsigned char* sourceImageData, int rows, int cols, unsigned char* detectionImageData, unsigned char* recognizedImageData
                   /*unsigned char*** alignedImagesData, unsigned int** alignedImagesWidth, unsigned int** alignedImagesHeight*/) {
    cv::Mat image(rows, cols, CV_8UC3, sourceImageData);
    auto size = image.size();
    const size_t width = size.width;
    const size_t height = size.height;

    // ---------------------------------------------------------------------------------------------------
    // --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
    std::string deviceName = "CPU";

    FaceDetection faceDetector(faceDetectionModel, deviceName, 1, false, false, 0.5, false);
    FacialLandmarksDetection facialLandmarksDetector(facialLandmarksModel, deviceName, 1, false, false);
    FeatureExtraction featureExtractor(featureExtractionModel, deviceName, 1, false, false);
    Classification classifier;

    InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
    // Disable dynamic batching for face detector as it processes one image at a time
    // Disable dynamic batching for feature extractor for prototype
    Load<decltype(faceDetector)>(faceDetector).into(plugin, false);
    Load<decltype(facialLandmarksDetector)>(facialLandmarksDetector).into(plugin, false);
    Load<decltype(featureExtractor)>(featureExtractor).into(plugin, false);
    // ----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Doing inference -----------------------------------------------------
    // Starting inference & calculating performance

    bool isFaceAnalyticsEnabled = facialLandmarksDetector.enabled();

    Timer timer;
    timer.start("total");

    std::ostringstream out;

    {
        // Detecting all faces on the first frame and reading the next one
        timer.start("detection");
        faceDetector.enqueue(image);
        faceDetector.submitRequest();
        faceDetector.wait();
        faceDetector.fetchResults();
        auto detectionResults = faceDetector.results;
        timer.finish("detection");

        std::vector<cv::Mat> detectedFaces;
        timer.start("data postprocessing");
        // Filling inputs of face analytics networks
        for (auto &&face : detectionResults) {
            if (isFaceAnalyticsEnabled) {
                auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                cv::Mat face = image(clippedRect);
                detectedFaces.push_back(face);
                facialLandmarksDetector.enqueue(face);
            }
        }
        timer.finish("data postprocessing");

        // Running Facial Landmarks Estimation network
        timer.start("facial landmarks detector");
        if (isFaceAnalyticsEnabled) {
            facialLandmarksDetector.submitRequest();
            facialLandmarksDetector.wait();
        }
        timer.finish("facial landmarks detector");

        timer.start("face preprocessing");
        std::vector<cv::Mat> alignedFaces;
        if (isFaceAnalyticsEnabled) {
            int i = 0;
            for (auto &result : detectionResults) {
                cv::Rect rect = result.location;
                auto normedLandmarks = facialLandmarksDetector[i];
                auto leftEye = { cv::Point2f { normedLandmarks[0], normedLandmarks[1] },
                                  cv::Point2f { normedLandmarks[2], normedLandmarks[3] } };
                auto rightEye = { cv::Point2f { normedLandmarks[4], normedLandmarks[5] },
                                   cv::Point2f { normedLandmarks[6], normedLandmarks[7] } };
                cv::Mat alignedFace = alignFace(detectedFaces[i], leftEye, rightEye);
                alignedFaces.push_back(alignedFace);

                if (!alignedFace.empty()) {
                    featureExtractor.enqueue(alignedFace);
                }

                ++i;
            }
        }
        timer.finish("face preprocessing");

        timer.start("feature extractor");
        std::vector<std::vector<float>> featureVectors;

        if (isFaceAnalyticsEnabled) {
            featureExtractor.submitRequest();
            featureExtractor.wait();
            featureExtractor.fetchResults();

            auto resultsSize = detectionResults.size();
            for (int i = 0; i < resultsSize; ++i) {
                featureVectors.push_back(featureExtractor.results);
            }
        }
        timer.finish("feature extractor");

        timer.start("classifier");

        std::vector<std::string> persons;
        for (auto featureVector : featureVectors) {
            auto label = classifier.classify(featureVector);
            persons.push_back(label);
        }
        timer.finish("classifier");

        // Visualizing results
        {
            timer.start("visualization");
//            out.str("");
//            out << "OpenCV render time: " << std::fixed << std::setprecision(2)
//                <<  timer["visualization"].getSmoothedDuration()
//                << " ms";
//            cv::putText(image, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
//                        cv::Scalar(255, 0, 0));

//            out.str("");
//            out << "Face detection time: " << std::fixed << std::setprecision(2)
//                << timer["detection"].getSmoothedDuration()
//                << " ms ("
//                << 1000.f / (timer["detection"].getSmoothedDuration())
//                << " fps)";
//            cv::putText(image, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
//                        cv::Scalar(255, 0, 0));

//            if (isFaceAnalyticsEnabled) {
//                out.str("");
//                out << "Facial Landmarks Detector Networks "
//                    << "time: " << std::fixed << std::setprecision(2)
//                    << timer["facial landmarks detector"].getSmoothedDuration()
//                    << " ms ";
//                if (!detectionResults.empty()) {
//                    out << "("
//                        << 1000.f / timer["facial landmarks detector"].getSmoothedDuration()
//                        << " fps)";
//                }
//                cv::putText(image, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5,
//                            cv::Scalar(255, 0, 0));

//                out.str("");
//                out << "Face preprocessing before feature extraction "
//                    << "time: " << std::fixed << std::setprecision(2)
//                    << timer["face preprocessing"].getSmoothedDuration() +
//                       timer["face preprocessing"].getSmoothedDuration()
//                    << " ms ";
//                cv::putText(image, out.str(), cv::Point2f(0, 85), cv::FONT_HERSHEY_TRIPLEX, 0.5,
//                            cv::Scalar(255, 0, 0));
//            }

            // For every detected face
            cv::Mat detectedFaces(rows, cols, CV_8UC3, detectionImageData);
            detectedFaces = image;
            cv::Mat recognizedFaces(rows, cols, CV_8UC3, recognizedImageData);
            recognizedFaces = image;

            int i = 0;
            for (auto &result : detectionResults) {

                cv::Rect rect = result.location;

                cv::rectangle(detectedFaces, result.location, cv::Scalar(100, 100, 100), 1);
                cv::rectangle(recognizedFaces, result.location, cv::Scalar(100, 100, 100), 1);

                out.str("");


                out << persons[i]
                    << ": " << std::fixed << std::setprecision(3) << result.confidence;


                cv::putText(recognizedFaces,
                            out.str(),
                            cv::Point2f(result.location.x, result.location.y - 15),
                            cv::FONT_HERSHEY_COMPLEX_SMALL,
                            0.8,
                            cv::Scalar(0, 0, 255));
                i++;
            }

            timer.finish("visualization");

//            for (auto alignedFace : alignedFaces) {
//            }
        }

        cv::waitKey(0);
        timer.finish("total");
    }

    return true;
}

int main(int argc, char *argv[]) {
    try {
            std::string path = retrievePath(argc, argv);
            if (path == "") {
                throw std::logic_error("Command line option with the path to the image is required.");
            }
            cv::Mat image = cv::imread(path);
            if (image.empty()) {
                throw std::logic_error("Incorrect path to the input image!");
            }

            unsigned char* detectionImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            unsigned char * recognizedImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            recognizeFace(image.data, image.size().width, image.size().height, detectionImageData, recognizedImageData);

        }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
