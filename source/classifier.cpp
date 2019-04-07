#include <cmath>

#include "classifier.hpp"
#include "tmp_database.hpp"

std::string  Classifier::classify(std::vector<float> featureVector) {
    std::map<std::string, float> minimums;
    for (auto it = classifiedFaces.begin(); it != classifiedFaces.end(); ++it) {
        auto itFeatureVectors = it->second;
        float min = 1000000;
        for (auto itClassifiedFeatureVector = itFeatureVectors.begin(); itClassifiedFeatureVector != itFeatureVectors.end(); ++itClassifiedFeatureVector) {
            auto classifiedFeatures = *itClassifiedFeatureVector;

            slog::info << "size: " << featureVector.size() << slog::endl;
//            if ((classifiedFeatures.size() != featureVector.size())) {
//               // throw std::logic_error("Classified feature vector size does not equal to input feature vector size!");
//            }

            std::vector<float> diffVector(featureVector.size());
            for (auto i = 0; i  < classifiedFeatures.size(); ++i) {
                diffVector[i] = featureVector[i] - classifiedFeatures[i];
            }

            float sum = 0.f;
            for (auto i = 0; i  < diffVector.size(); ++i) {
                sum += diffVector[i] * diffVector[i];
            }

            sum = sqrtf(sum);
            if (sum < min) {
                min = sum;
            }
        }
       minimums[it->first] = min;
    }
    
    float globalMin = 1000000;
    auto globalMinIt = minimums.begin();
    for ( auto minIt = globalMinIt; minIt != minimums.end(); ++minIt) {
       if (minIt->second < globalMin) {
           globalMin = minIt->second;
           globalMinIt = minIt;
       }
    }
    slog::info << globalMinIt->first << " : " << globalMinIt->second << slog::endl;

    return  globalMinIt->first;
}
