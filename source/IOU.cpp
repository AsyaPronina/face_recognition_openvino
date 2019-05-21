#include <IOU.hpp>

double calculateIOU(cv::Rect predictedBox, cv::Rect groundTruthBox) {

        int xA = std::max(predictedBox.x, groundTruthBox.x);
        int yA = std::max(predictedBox.y, groundTruthBox.y);

        int xB = std::min(predictedBox.x + predictedBox.width, groundTruthBox.x + groundTruthBox.width);
        int yB = std::min(predictedBox.y + predictedBox.height, groundTruthBox.y + groundTruthBox.height);

        int interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);

        //slog::info << "interArea: " << interArea << slog::endl;

        int boxAArea = predictedBox.width * predictedBox.height;
        //slog::info << "boxAArea: " << boxAArea << slog::endl;
        int boxBArea = groundTruthBox.width * groundTruthBox.height;
        //slog::info << "boxBArea: " << boxBArea << slog::endl;

        double IOU = interArea / double(boxAArea + boxBArea - interArea);

        return IOU;
}
