# pragma once

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

#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

struct FeatureExtraction {
    InferenceEngine::ExecutableNetwork net;
    InferenceEngine::InferencePlugin * plugin;
    InferenceEngine::InferRequest::Ptr request;
    std::string topoName;
    std::string pathToModel;
    std::string deviceForInference;
    const int maxBatch;
    bool isBatchDynamic;
    const bool isAsync;
    mutable bool enablingChecked;
    mutable bool _enabled;

    FeatureExtraction(const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync);

    virtual ~FeatureExtraction();

    bool enabled() const;

    InferenceEngine::ExecutableNetwork* operator ->();
    virtual InferenceEngine::CNNNetwork read();
    void enqueue(const cv::Mat &frame);
    virtual void submitRequest();
    virtual void wait();
    void fetchResults();
    void printPerformanceCounts();

    std::string input;
    std::string output;
    int enquedFrames;
    float width;
    float height;
    bool resultsFetched;
    int featureVectorSize;
    std::vector<float> results;
};
