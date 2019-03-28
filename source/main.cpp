// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <gflags/gflags.h>
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

#include "detectors.hpp"

#include <ie_iextension.h>
#include <ext_list.hpp>

DEFINE_string(m, "/home/asya/Study/DL/face_recognition/models/face-detection-adas-0001.xml", face_detection_model_message);
DEFINE_string(m_lm, "", facial_landmarks_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_lm, "CPU", target_device_message_lm);
DEFINE_uint32(n_lm, 16, num_batch_em_message);
DEFINE_bool(dyn_lm, false, dyn_batch_em_message);
/// @brief Define parameter for GPU custom kernels path<br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);
/// \brief Define a flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);
/// \brief Define a parameter for probability threshold for detections<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_bool(no_wait, false, no_wait_for_keypress_message);
DEFINE_bool(async, false, async_message);


using namespace InferenceEngine;

int main(int argc, char *argv[]) {
    try {
            cv::VideoCapture cap;
            if (!cap.open(0)) {
                throw std::logic_error("Cannot open camera: ");
            }
            const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
            const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

            // read input (video) frame
            cv::Mat frame;
            if (!cap.read(frame)) {
                throw std::logic_error("Failed to get frame from cv::VideoCapture");
            }
            
            // ---------------------------------------------------------------------------------------------------
            // --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
            std::map<std::string, InferencePlugin> pluginsForDevices;
            std::vector<std::pair<std::string, std::string>> cmdOptions = {
                {FLAGS_d, FLAGS_m}, {FLAGS_d_lm, FLAGS_m_lm}
            };
            FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
            FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async);

            std::string deviceName = "CPU";
            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                    continue;
            }
            
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Loading extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

               }
               pluginsForDevices[deviceName] = plugin;

            // ---------------------------------------------------------------------------------------------------
    
            // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
            // Disable dynamic batching for face detector as it processes one image at a time
            Load(faceDetector).into(pluginsForDevices[FLAGS_d], false);
            Load(facialLandmarksDetector).into(pluginsForDevices[FLAGS_d_lm], FLAGS_dyn_lm);
            // ----------------------------------------------------------------------------------------------------
    
            // --------------------------- 3. Doing inference -----------------------------------------------------
            // Starting inference & calculating performance
    
            bool isFaceAnalyticsEnabled = facialLandmarksDetector.enabled();
    
            Timer timer;
            timer.start("total");
    
            std::ostringstream out;
            size_t framesCounter = 0;
            bool frameReadStatus;
            bool isLastFrame;
            cv::Mat prev_frame, next_frame;
    
            // Detecting all faces on the first frame and reading the next one
            timer.start("detection");
            faceDetector.enqueue(frame);
            faceDetector.submitRequest();
            timer.finish("detection");
    
            prev_frame = frame.clone();
    
            // Reading the next frame
            timer.start("video frame decoding");
            frameReadStatus = cap.read(frame);
            timer.finish("video frame decoding");
    
            while (true) {
                framesCounter++;
                isLastFrame = !frameReadStatus;
    
                timer.start("detection");
                // Retrieving face detection results for the previous frame
                faceDetector.wait();
                faceDetector.fetchResults();
                auto prev_detection_results = faceDetector.results;
    
                // No valid frame to infer if previous frame is the last
                if (!isLastFrame) {
                    faceDetector.enqueue(frame);
                    faceDetector.submitRequest();
                }
                timer.finish("detection");
    
                timer.start("data postprocessing");
                // Filling inputs of face analytics networks
                for (auto &&face : prev_detection_results) {
                    if (isFaceAnalyticsEnabled) {
                        auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                        cv::Mat face = prev_frame(clippedRect);
                        facialLandmarksDetector.enqueue(face);
                    }
                }
                timer.finish("data postprocessing");
    
                // Running Facial Landmarks Estimation network
                timer.start("facial landmarks detector");
                if (isFaceAnalyticsEnabled) {
                    facialLandmarksDetector.submitRequest();
                }
                timer.finish("facial landmarks detector");
    
                // Reading the next frame if the current one is not the last
                if (!isLastFrame) {
                    timer.start("video frame decoding");
                    frameReadStatus = cap.read(next_frame);
                    timer.finish("video frame decoding");
                }
    
                timer.start("facial landmarks detector wait");
                if (isFaceAnalyticsEnabled) {
                    facialLandmarksDetector.wait();
                }
                timer.finish("facial landmarks detector wait");
    
                // Visualizing results
                {
                    timer.start("visualization");
                    out.str("");
                    out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                        << (timer["video frame decoding"].getSmoothedDuration() +
                           timer["visualization"].getSmoothedDuration())
                        << " ms";
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
    
                    out.str("");
                    out << "Face detection time: " << std::fixed << std::setprecision(2)
                        << timer["detection"].getSmoothedDuration()
                        << " ms ("
                        << 1000.f / (timer["detection"].getSmoothedDuration())
                        << " fps)";
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
    
                    if (isFaceAnalyticsEnabled) {
                        out.str("");
                        out << "Facial Landmarks Detector Networks "
                            << "time: " << std::fixed << std::setprecision(2)
                            << timer["facial landmarks detector call"].getSmoothedDuration() +
                               timer["face landmarks detector wait"].getSmoothedDuration()
                            << " ms ";
                        if (!prev_detection_results.empty()) {
                            out << "("
                                << 1000.f / (timer["facial landmarks detector call"].getSmoothedDuration() +
                                   timer["facial landmarks detector wait"].getSmoothedDuration())
                                << " fps)";
                        }
                        cv::putText(prev_frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                    cv::Scalar(255, 0, 0));
                    }
    
                    // For every detected face
                    int i = 0;
                    for (auto &result : prev_detection_results) {
                        cv::Rect rect = result.location;
    
                        out.str("");
    
                        
                        out << (result.label < faceDetector.labels.size() ? faceDetector.labels[result.label] :
                                std::string("label #") + std::to_string(result.label))
                            << ": " << std::fixed << std::setprecision(3) << result.confidence;
    
    
                        cv::putText(prev_frame,
                                    out.str(),
                                    cv::Point2f(result.location.x, result.location.y - 15),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.8,
                                    cv::Scalar(0, 0, 255));
    
                        if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
                            auto normed_landmarks = facialLandmarksDetector[i];
                            auto n_lm = normed_landmarks.size();
                            if (FLAGS_r)
                                std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
                            for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                                float normed_x = normed_landmarks[2 * i_lm];
                                float normed_y = normed_landmarks[2 * i_lm + 1];
    
                                if (FLAGS_r) {
                                    std::cout << normed_x << ", "
                                              << normed_y << std::endl;
                                }
                                int x_lm = rect.x + rect.width * normed_x;
                                int y_lm = rect.y + rect.height * normed_y;
                                // Drawing facial landmarks on the frame
                                cv::circle(prev_frame, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
                            }
                        }
    
                        cv::rectangle(prev_frame, result.location, cv::Scalar(100, 100, 100), 1);
                        i++;
                    }
    
                    cv::imshow("Detection results", prev_frame);
                    timer.finish("visualization");
                }
    
                // End of file (or a single frame file like an image). The last frame is displayed to let you check what is shown
                if (isLastFrame) {
                    timer.finish("total");
                    if (!FLAGS_no_wait) {
                        std::cout << "No more frames to process. Press any key to exit" << std::endl;
                        cv::waitKey(0);
                    }
                    break;
                } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                    timer.finish("total");
                    break;
                }
    
                prev_frame = frame;
                frame = next_frame;
                next_frame = cv::Mat();
            }
    
            slog::info << "Number of processed frames: " << framesCounter << slog::endl;
            slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

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
