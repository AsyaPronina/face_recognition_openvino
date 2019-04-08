#include <cstdint>

#include "feature_extractor.hpp"

FeatureExtraction::FeatureExtraction(const std::string &pathToModel,
                             const std::string &deviceForInference,
                             int maxBatch, bool isBatchDynamic, bool isAsync)
    : topoName("Feature extraction"), pathToModel(pathToModel), deviceForInference(deviceForInference),
      maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
      enablingChecked(false), _enabled(false), enquedFrames(0), width(0), height(0), resultsFetched(false) {
    if (isAsync) {
        slog::info << "Use async mode for " << topoName << slog::endl;
    }
}

FeatureExtraction::~FeatureExtraction() {}

InferenceEngine::ExecutableNetwork* FeatureExtraction::operator ->() {
    return &net;
}

bool FeatureExtraction::enabled() const  {
    if (!enablingChecked) {
        _enabled = !pathToModel.empty();
        if (!_enabled) {
            slog::info << topoName << " DISABLED" << slog::endl;
        }
        enablingChecked = true;
    }
    return _enabled;
}

void FeatureExtraction::enqueue(const cv::Mat &frame) {
    if (!enabled()) return;

    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    width = frame.cols;
    height = frame.rows;

    InferenceEngine::Blob::Ptr  inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enquedFrames = 1;
}

void FeatureExtraction::submitRequest() {
    if (!enquedFrames) return;
    enquedFrames = 0;
    resultsFetched = false;
    results.clear();

    if (!enabled() || request == nullptr) return;
    if (isAsync) {
        request->StartAsync();
    } else {
        request->Infer();
    }
}

void FeatureExtraction::wait() {
    if (!enabled()|| !request || !isAsync)
        return;
    request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

InferenceEngine::CNNNetwork FeatureExtraction::read()  {
    slog::info << "Loading network files for Feature Extractor" << slog::endl;
    InferenceEngine::CNNNetReader netReader;
    /** Read network model **/
    netReader.ReadNetwork(pathToModel);
    /** Set batch size to 1 **/
    slog::info << "Batch size is set to " << maxBatch << slog::endl;
    netReader.getNetwork().setBatchSize(maxBatch);
    /** Extract model name and load its weights **/
    std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
    netReader.ReadWeights(binFileName);
   // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check inputs -------------------------------------------------------------
    slog::info << "Checking Feature Extractor network inputs" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error(std::to_string(inputInfo.size()) + " Feature Extractor network should have only one input");
    }
    InferenceEngine::InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);

    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    slog::info << "Checking Feature Extractor network outputs" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error(std::to_string(outputInfo.size()) + "Face Detection network should have only one output");
    }
    InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
    output = outputInfo.begin()->first;

    const InferenceEngine::CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
    if (outputLayer->type != "FullyConnected") {
        throw std::logic_error("Feature Extractor network output layer(" + outputLayer->name +
                               ") should be DetectionOutput, but was " +  outputLayer->type);
    }

    if (outputLayer->params.find("out-size") == outputLayer->params.end()) {
        throw std::logic_error("Feature Extractor network output layer (" +
                               output + ") should have out-size integer attribute");
    }

    const int outSize = outputLayer->GetParamAsInt("out-size");
    if (outSize != 512) {
        throw std::logic_error("Feature Extraction networkout-size integer attribute shall be equal to 512");
    }

    const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
    featureVectorSize = outputDims[1];
    if (featureVectorSize != outSize) {
        throw std::logic_error("Feature Extraction network output layer should have 512 as a last dimension");
    }
    if (outputDims.size() != 2) {
        throw std::logic_error("Feature Extraction network output dimensions not compatible shoulld be 2, but was " +
                               std::to_string(outputDims.size()));
    }
    _output->setPrecision(InferenceEngine::Precision::FP32);

    slog::info << "Loading Feature Extraction model to the "<< deviceForInference << " plugin" << slog::endl;
    input = inputInfo.begin()->first;
    return netReader.getNetwork();
}

void FeatureExtraction::fetchResults() {
    if (!enabled()) return;
    results.clear();
    if (resultsFetched) return;
    resultsFetched = true;
    const float *featureVector = request->GetBlob(output)->buffer().as<float *>();

    for (int i = 0; i < featureVectorSize; ++i) {
        results.push_back(featureVector[i]);
        slog::info << featureVector[i] << ' ';
    }
    slog::info << slog::endl;
}

void FeatureExtraction::printPerformanceCounts() {
    if (!enabled()) {
        return;
    }
    slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
    ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
}
