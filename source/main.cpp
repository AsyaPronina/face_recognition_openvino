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

//Switch to singletone
Timer timer;
std::vector<cv::Mat> alignedFaces;

std::string faceDetectionModel = "models/face-detection-adas-0001.xml";
std::string facialLandmarksModel = "models/facial-landmarks-35-adas-0001.xml";
std::string featureExtractionModel = "models/Sphereface.xml";

using namespace InferenceEngine;

std::string retrievePath(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    if (argc == 2)
    {
        return argv[1];
    }

    return "";
}

extern "C" void clear() {
    alignedFaces.clear();
}

extern "C" double getFaceRecognitionTime() {
    return timer["total"].getSmoothedDuration();
}

extern "C" int getAlignedFacesCount() {
     return alignedFaces.size();
}

extern "C" void getAlignedFacesSizes(unsigned int* widthData, unsigned int* heightData) {
    for (auto alignedFace : alignedFaces) {
        *widthData = alignedFace.size().width;
        *heightData = alignedFace.size().height;

        ++widthData;
        ++heightData;
    }
}

extern "C" void getAlignedFaces(unsigned char* alignedImagesData) {
    for (auto alignedFace : alignedFaces) {
        auto width = alignedFace.size().width;
        auto height = alignedFace.size().height;

        cv::Mat alignedImageMat(height, width, CV_8UC3, alignedImagesData);
        alignedFace.copyTo(alignedImageMat);

        alignedImagesData += width * height * 3;
    }
}

extern "C" void recognizeFaces(unsigned char* sourceImageData, int rows, int cols, unsigned char* detectionImageData, unsigned char* recognizedImageData, char* pathCalcMAP, char* pathRecognitionResult) {
    cv::Mat image(rows, cols, CV_8UC3, sourceImageData);

    auto size = image.size();
    const size_t width = size.width;
    const size_t height = size.height;

    // ---------------------------------------------------------------------------------------------------
    // --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
    std::string deviceName = "CPU";
    std::map<std::string, InferencePlugin> pluginsForDevices;

    FaceDetection faceDetector(faceDetectionModel, deviceName, 1, false, false, 0.5, false);
    FacialLandmarksDetection facialLandmarksDetector(facialLandmarksModel, deviceName, 16, false, false);
    FeatureExtraction featureExtractor(featureExtractionModel, deviceName, 1, false, false);
    Classification classifier;

    InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    pluginsForDevices[deviceName] = plugin;

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
    // Disable dynamic batching for face detector as it processes one image at a time
    // Disable dynamic batching for feature extractor for prototype
    Load<decltype(faceDetector)>(faceDetector).into(pluginsForDevices[deviceName], false);
    Load<decltype(facialLandmarksDetector)>(facialLandmarksDetector).into(pluginsForDevices[deviceName], false);
    Load<decltype(featureExtractor)>(featureExtractor).into(pluginsForDevices[deviceName], false);
    // ----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Doing inference -----------------------------------------------------
    // Starting inference & calculating performance

    bool isFaceAnalyticsEnabled = facialLandmarksDetector.enabled();

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
                alignedFaces.push_back(alignedFace); // alignedFaces is temporarily global variable

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
        timer.finish("total");

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
            image.copyTo(detectedFaces);
            cv::Mat recognizedFaces(rows, cols, CV_8UC3, recognizedImageData);
            image.copyTo(recognizedFaces);

            int i = 0;
            for (auto &result : detectionResults) {

                cv::Rect rect = result.location;

                // resize rect according center
                double x_off = result.location.width*0.2;
                double y_off = result.location.height*0.15;
                result.location.x=int(result.location.x+x_off);
                result.location.y=int(result.location.y+y_off * 1.7);
                result.location.width=int(result.location.width-1.9* x_off);
                result.location.height=int(result.location.height-2.5 * y_off);

                cv::rectangle(detectedFaces, result.location, cv::Scalar(100, 100, 100), 1);
                cv::rectangle(recognizedFaces, result.location, cv::Scalar(100, 100, 100), 1);

                out.str("");


                out << persons[i];
                    //Here is detection confidence, but recognition confidence shall be calculated.
                    //<< ": " << std::fixed << std::setprecision(3) << result.confidence;


                cv::putText(recognizedFaces,
                            out.str(),
                            cv::Point2f(result.location.x, result.location.y - 15),
                            cv::FONT_HERSHEY_COMPLEX_SMALL,
                            0.8,
                            cv::Scalar(0, 0, 255));
                i++;
            }

            timer.finish("visualization");
        }

        //saving to file results of recognition and detection for metrics (coords of bbox,class)
         int j = 0;
        //slog::info << "check path" <<pathCalcMAP<< slog::endl <<pathRecognitionResult<< slog::endl;
         std::ofstream outfile_predicted(pathCalcMAP);
         std::ofstream outfile_with_classes(pathRecognitionResult);
         for (auto &result : detectionResults) {
             outfile_predicted << "detection" << " " << result.confidence<< " " << result.location.x << " " << result.location.y << " " << result.location.x+result.location.width << " " << result.location.y+result.location.height << std::endl;
             outfile_with_classes << persons[j] << " " << result.confidence<< " " << result.location.x << " " << result.location.y << " " << result.location.x+result.location.width << " " << result.location.y+result.location.height << std::endl;
             j++;
         }
         outfile_predicted.close();
         outfile_with_classes.close();
    }
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

            char pathCalcMAP[path.length()+1];
            char pathRecognitionResult[path.length()+1];
            //ONLY TO DEBUG
            unsigned char* detectionImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            unsigned char* recognizedImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            recognizeFaces(image.data, image.size().height, image.size().width, detectionImageData, recognizedImageData, pathCalcMAP, pathRecognitionResult);
            slog::info << "Crash will no be output here" << slog::endl;

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
