#include <cstdint>

#include "classifier_model.hpp"

std::vector<std::string> ClassificationModel::classes{ "Asyok", "daryafret", "Nastya", "Malinka", "Ion", "Unknown" };

ClassificationModel::ClassificationModel(const std::string &pathToModel,
	const std::string &deviceForInference,
	int maxBatch, bool isBatchDynamic, bool isAsync)
	: topoName("4 FC layers - classification net"), pathToModel(pathToModel), deviceForInference(deviceForInference),
	maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
	enablingChecked(false), _enabled(false), enquedFrames(0), width(0), height(0), resultsFetched(false) {
	if (isAsync) {
		slog::info << "Use async mode for " << topoName << slog::endl;
	}
}

ClassificationModel::~ClassificationModel() {
	request.reset();
}

InferenceEngine::ExecutableNetwork* ClassificationModel::operator ->() {
	return &net;
}

bool ClassificationModel::enabled() const {
	if (!enablingChecked) {
		_enabled = !pathToModel.empty();
		if (!_enabled) {
			slog::info << topoName << " DISABLED" << slog::endl;
		}
		enablingChecked = true;
	}
	return _enabled;
}

void ClassificationModel::enqueue(const std::vector<float>& featureVector) {
	if (!enabled()) return;

	if (!request) {
		request = net.CreateInferRequestPtr();
	}

	width = featureVector.size();
	height = 1;

	InferenceEngine::Blob::Ptr  inputBlob = request->GetBlob(input);

	InferenceEngine::SizeVector blobSize = inputBlob->getTensorDesc().getDims();
	const size_t blobWidth = blobSize[1];
	assert(blobWidth == width);

	float* blob_data = inputBlob->buffer().as<float*>();

	for (size_t bw = 0; bw < blobWidth; ++bw) {
		blob_data[bw] = featureVector[bw];
	}

	enquedFrames = 1;
}

void ClassificationModel::submitRequest() {
	if (!enquedFrames) return;
	enquedFrames = 0;
	resultsFetched = false;
	result.clear();

	if (!enabled() || request == nullptr) return;
	if (isAsync) {
		request->StartAsync();
	}
	else {
		request->Infer();
	}
}

void ClassificationModel::wait() {
	if (!enabled() || !request || !isAsync)
		return;
	request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

InferenceEngine::CNNNetwork ClassificationModel::read() {
	slog::info << "Loading network files for Classification" << slog::endl;
	InferenceEngine::CNNNetReader netReader;
	/** Read network model **/
	netReader.ReadNetwork(pathToModel);
	/** Set batch size **/
	slog::info << "Batch size is set to " << maxBatch << slog::endl;
	netReader.getNetwork().setBatchSize(maxBatch);
	/** Extract model name and load its weights **/
	std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
	netReader.ReadWeights(binFileName);
	// -----------------------------------------------------------------------------------------------------

	// ---------------------------Check inputs -------------------------------------------------------------
	slog::info << "Checking Classification network inputs" << slog::endl;
	InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
	if (inputInfo.size() != 1) {
		throw std::logic_error(std::to_string(inputInfo.size()) + " Classification network should have only one input");
	}
	InferenceEngine::InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
	inputInfoFirst->setPrecision(InferenceEngine::Precision::FP32);

	// -----------------------------------------------------------------------------------------------------

	// ---------------------------Check outputs ------------------------------------------------------------
	slog::info << "Checking Classification network outputs" << slog::endl;
	InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
	if (outputInfo.size() != 1) {
		throw std::logic_error(std::to_string(outputInfo.size()) + "Classification network should have only one output");
	}
	InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
	output = outputInfo.begin()->first;

	const InferenceEngine::CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());

	if (outputLayer->type != "SoftMax") {
		slog::info << "Classification network output layer(" + outputLayer->name +
			") should be SoftMax, but was " + outputLayer->type;
		throw std::logic_error("Classification network output layer(" + outputLayer->name +
			") should be SoftMax, but was " + outputLayer->type);
	}

	if (outputLayer->params.find("axis") == outputLayer->params.end()) {
		slog::info << "Classification network output layer(" +
			output + ") should have axis integer attribute";
		throw std::logic_error("Classification network output layer(" +
			output + ") should have axis integer attribute");
	}

	const int axis = outputLayer->GetParamAsInt("axis");
	if (axis != 1) {
		slog::info << "Classification network axis integer attribute shall be equal to 1";
		throw std::logic_error("Classification network axis integer attribute shall be equal to 1");
	}

	const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
	probabilityVectorSize = outputDims[1];
	if (probabilityVectorSize != 6) {
		slog::info << "Classification network output layer should have 6 as a last dimension";
		throw std::logic_error("Classification network output layer should have 6 as a last dimension");
	}
	if (outputDims.size() != 2) {
		slog::info << "Classification network output dimensions not compatible shoulld be 2, but was " +
			std::to_string(outputDims.size());
		throw std::logic_error("Classification network output dimensions not compatible shoulld be 2, but was " +
			std::to_string(outputDims.size()));
	}
	_output->setPrecision(InferenceEngine::Precision::FP32);

	slog::info << "Loading Classification model to the " << deviceForInference << " plugin" << slog::endl;
	input = inputInfo.begin()->first;
	return netReader.getNetwork();
}

void ClassificationModel::fetchResults() {
	if (!enabled()) return;
	result.clear();
	if (resultsFetched) return;
	resultsFetched = true;
	const float *probabilityVector = request->GetBlob(output)->buffer().as<float *>();

	slog::info << "ProbabilityVector: ";

	int maxI = 0;
	float max = 0.f;
	for (int i = maxI; i < probabilityVectorSize; ++i) {
		if (probabilityVector[i] > max) {
			max = probabilityVector[i];
			maxI = i;
		}
		slog::info << probabilityVector[i] << " ";
	}

	slog::info << slog::endl;

	result = classes[maxI];
}

void ClassificationModel::printPerformanceCounts() {
	if (!enabled()) {
		return;
	}
	slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
	::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
}
