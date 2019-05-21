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
#include "classifier_model.hpp"

#include "IOU.hpp"

#ifdef __linux__
    #define FR_EXPORT
#elif _WIN32
    #define FR_EXPORT __declspec(dllexport)
#endif

//add #ifdef linux, #else windows for __declspec(dllexpoert) as it is not portable
//Switch to singletone
Timer timer;
std::vector<cv::Mat> alignedFaces;
std::vector<cv::Mat> detectedFaces;

std::string faceDetectionModel = "models/face-detection-adas-0001.xml";
std::string facialLandmarksModel = "models/facial-landmarks-35-adas-0001.xml";
std::string featureExtractionModel = "models/Sphereface.xml";
std::string classificationModel = "models/classifier.xml";

std::string currentClass;

static std::unordered_map<std::string, std::vector<std::vector<float>>> facesFeatureVectors{ };

using namespace InferenceEngine;

std::string retrievePath(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    if (argc == 2)
    {
        return argv[1];
    }

    return "";
}

//--------For classifier train purposes:--------------
extern "C" FR_EXPORT void setCurrentClass(char* label) {
	currentClass = label;
}

extern "C" FR_EXPORT void clearCurrentClass() {
	currentClass = "";
}

extern "C" FR_EXPORT void dumpFeatureVectorsToJson(char* path) {
	std::ofstream featureVectorsJson(path);
	
	featureVectorsJson << "{\n  \"labels\":[ \"" << facesFeatureVectors.begin()->first << "\"";
	for (auto it = ++(facesFeatureVectors.begin()); it != facesFeatureVectors.end(); ++it) {
		featureVectorsJson << ", \"" << it->first << "\"";
	}
	featureVectorsJson << " ],\n";

	auto firstFaceIt = facesFeatureVectors.begin();
	featureVectorsJson << "  " << "\"" << firstFaceIt->first << "\":[";

	auto featureVectorsForClass = firstFaceIt->second;

	auto firstFeatureVector = *featureVectorsForClass.begin();
	featureVectorsJson << "\"" << firstFeatureVector[0];
	for (int i = 1; i < firstFeatureVector.size(); ++i) {
		featureVectorsJson << ", " << firstFeatureVector[i];
	}
	featureVectorsJson << "\"";

	for (auto it1 = (++featureVectorsForClass.begin()); it1 != featureVectorsForClass.end(); ++it1) {
		auto featureVector = *it1;
		featureVectorsJson << ", \"" << featureVector[0];
		for (int i = 1; i < featureVector.size(); ++i) {
			featureVectorsJson << ", " << featureVector[i];
		}
		featureVectorsJson << "\"";
	}

	featureVectorsJson << "]";

	for (auto it = (++facesFeatureVectors.begin()); it != facesFeatureVectors.end(); ++it) {
		featureVectorsJson << ",\n  " << "\"" << it->first << "\":[";

		auto featureVectorsForClass = it->second;

		auto firstFeatureVector = *featureVectorsForClass.begin();
		featureVectorsJson << "\"" << firstFeatureVector[0];
		for (int i = 1; i < firstFeatureVector.size(); ++i) {
			featureVectorsJson << ", " << firstFeatureVector[i];
		}
		featureVectorsJson << "\"";

		for (auto it1 = (++featureVectorsForClass.begin()); it1 != featureVectorsForClass.end(); ++it1) {
			auto featureVector = *it1;
			featureVectorsJson << ", \"" << featureVector[0];
			for (int i = 1; i < featureVector.size(); ++i) {
				featureVectorsJson << ", " << featureVector[i];
			}
			featureVectorsJson << "\"";
		}

		featureVectorsJson << "]";
	}

	featureVectorsJson << "}";
}
//----------------------------------------------------


extern "C" FR_EXPORT void clear() {
    alignedFaces.clear();
    detectedFaces.clear();
}


extern "C" FR_EXPORT double getFaceRecognitionTime() {
    return timer["total"].getSmoothedDuration();
}

extern "C" FR_EXPORT int getAlignedFacesCount() {
     return alignedFaces.size();
}

extern "C" FR_EXPORT void getAlignedFacesSizes(unsigned int* widthData, unsigned int* heightData) {
    for (auto alignedFace : alignedFaces) {
        *widthData = alignedFace.size().width;
        *heightData = alignedFace.size().height;

        ++widthData;
        ++heightData;
    }
}

extern "C" FR_EXPORT void getAlignedFaces(unsigned char* alignedImagesData) {
    for (auto alignedFace : alignedFaces) {
        auto width = alignedFace.size().width;
        auto height = alignedFace.size().height;

        cv::Mat alignedImageMat(height, width, CV_8UC3, alignedImagesData);
        alignedFace.copyTo(alignedImageMat);

        alignedImagesData += width * height * 3;
    }
}

extern "C" FR_EXPORT void getDetectedFaces(unsigned char* detectedImagesData) {
    for (auto detectedFace : detectedFaces) {
        auto width = detectedFace.size().width;
        auto height = detectedFace.size().height;

        cv::Mat detectedImageMat(height, width, CV_8UC3, detectedImagesData);
        detectedFace.copyTo(detectedImageMat);

        detectedImagesData += width * height * 3;
    }
}

extern "C" FR_EXPORT void recognizeFaces(unsigned char* sourceImageData, int rows, int cols, unsigned char* detectionImageData, unsigned char* recognizedImageData,
                                         char* pathCalcMAP, char* pathRecognitionResult, int* groundTruth) {
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
	ClassificationModel classifierNet(classificationModel, deviceName, 1, false, false);

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
	Load<decltype(classifierNet)>(classifierNet).into(pluginsForDevices[deviceName], false);
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

        if (detectionResults.empty()) {
            return;
        }

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

        timer.start("face preprocessing && feature extractor");
        std::vector<std::vector<float>> featureVectors;

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
                    featureExtractor.submitRequest();
                    featureExtractor.wait();
                    featureExtractor.fetchResults();

                    featureVectors.push_back(featureExtractor.results);
                }

                ++i;
            }
        }
        timer.finish("face preprocessing && feature extractor");

        timer.start("classifier");

        for (auto i = 0; i < featureVectors.size(); ++i) {
             auto featueVector = featureVectors[i];

             auto label = classifier.classify(featueVector);
             detectionResults[i].label = label;

            /*classifierNet.enqueue(featureVector);
			classifierNet.submitRequest();
			classifierNet.wait();
			classifierNet.fetchResults();

			auto label = classifierNet.result;
            detectionResults[i].label = label;*/
        }

        timer.finish("classifier");
        timer.finish("total");

        for (auto &result : detectionResults) {

             // resize rect according center
            double x_off = result.location.width*0.2;
            double y_off = result.location.height*0.15;
            result.location.x = int(result.location.x+x_off);
            result.location.y = int(result.location.y+y_off * 1.7);
            result.location.width = int(result.location.width-1.9* x_off);
            result.location.height = int(result.location.height-2.5 * y_off);
        }

        if (groundTruth) {
            std::vector<FaceDetection::Result> groundTruthBoxes { };
            for (auto i = 0; i < 24; i += 4) {
                slog::info << groundTruth[i];

                bool isValid = true;
                for (auto j = 1; j < 4; ++j) {
                    slog::info << ", " << groundTruth[i + j];
                }

                slog::info << slog::endl;
            }

             for (auto i = 0; i < 24; i += 4) {

                if (groundTruth[i] != 0 || groundTruth[i + 2] != 0) {
                    int index = i / 4;
                    groundTruthBoxes.push_back(FaceDetection::Result { ClassificationModel::classes[index], 1.0, cv::Rect{ groundTruth[i], groundTruth[i + 1], groundTruth[i + 2], groundTruth[i + 3] } });
                }
             }

            for (auto j = 0; j < detectionResults.size(); ++j) {
                double maxIOU = 0.0;
                std::string searchedLabel = "";
                for (auto k = 0; k < groundTruthBoxes.size(); ++k) {
                    //slog::info << "detected box : " << detectionResults[j].location;
                    double IOU = calculateIOU(detectionResults[j].location, groundTruthBoxes[k].location);
                    //slog::endl;
                    if (IOU > maxIOU) {
                        maxIOU = IOU;
                        searchedLabel = groundTruthBoxes[k].label;
                    }
                }

                //slog::info << "maxIOU: " << maxIOU << slog::endl;
                if (maxIOU > 0.5) {
                    facesFeatureVectors[searchedLabel].push_back(featureVectors[j]);
                    //slog::info << searchedLabel << slog::endl;
                    //cv::imshow(searchedLabel, alignedFaces[j]);
                    //cv::waitKey();
                }
                else {
                    slog::info << "For detected box with label: " << detectionResults[j].label << " there is no ground truth box!" << slog::endl;
                    std::runtime_error(std::string("For detected box with label: ") + detectionResults[j].label + " there is no ground truth box!\n");
                }
            }
        }
        else {
            for (auto featureVector : featureVectors) {
                facesFeatureVectors[currentClass].push_back(featureVector);
            }
        }

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

            for (auto &result : detectionResults) {

                cv::rectangle(detectedFaces, result.location, cv::Scalar(100, 100, 100), 5);
                cv::rectangle(recognizedFaces, result.location, cv::Scalar(100, 100, 100), 5);

                out.str("");


                out << result.label;
                    //Here is detection confidence, but recognition confidence shall be calculated.
                    //<< ": " << std::fixed << std::setprecision(3) << result.confidence;


                cv::putText(recognizedFaces,
                            out.str(),
                            cv::Point2f(result.location.x, result.location.y - 15),
                            cv::FONT_HERSHEY_COMPLEX,
                            2,
                            cv::Scalar(0, 0, 255));
            }

            timer.finish("visualization");
        }

                //saving to file results of recognition and detection for metrics (coords of bbox,class)
         slog::info << "check path" <<pathCalcMAP<< slog::endl <<pathRecognitionResult<< slog::endl;
         std::ofstream outfile_predicted(pathCalcMAP);
         std::ofstream outfile_with_classes(pathRecognitionResult);
         for (auto &result : detectionResults) {
             outfile_predicted << "detection" << " " << result.confidence<< " " << result.location.x << " " << result.location.y << " " << result.location.x+result.location.width << " " << result.location.y+result.location.height << std::endl;
             outfile_with_classes << result.label << " " << result.confidence<< " " << result.location.x << " " << result.location.y << " " << result.location.x+result.location.width << " " << result.location.y+result.location.height << std::endl;
         }
         outfile_predicted.close();
         outfile_with_classes.close();

         slog::info << "Database:" << slog::endl;
		 for (auto it = facesFeatureVectors.begin(); it != facesFeatureVectors.end(); ++it) {
			 slog::info << it->first << ": ";
			 auto featuresVectorsForFace = it->second;

			 for (auto it1 = featuresVectorsForFace.begin(); it1 != featuresVectorsForFace.end(); ++it1) {
				 auto featuresVectorForFace = *it1;

				 slog::info << featuresVectorForFace[0] << ", " << featuresVectorForFace[1] << "..." << slog::endl;
			 }

			 slog::info << "End of feature vectors for: " << it->first << slog::endl << slog::endl;
         }

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

            //char pathCalcMAP[path.length()+1];
            //char pathRecognitionResult[path.length()+1];

            //variable-length arrays are extensions of gcc, so for Windows cl:
            char pathCalcMAP[100];
            char pathRecognitionResult[100];

            //ONLY TO DEBUG
            unsigned char* detectionImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            unsigned char* recognizedImageData = static_cast<unsigned char*>(malloc(image.size().width * image.size().height * image.depth() * sizeof(unsigned char)));
            //recognizeFaces(image.data, image.size().height, image.size().width);
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
