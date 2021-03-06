#include <cmath>

#include "classifier.hpp"
#include "tmp_database.hpp"

//Fin angle, think about threshold.
std::string  Classification::classify(std::vector<float> featureVector) {
    std::map<std::string, float> minimums;
    for (auto it = classifiedFaces.begin(); it != classifiedFaces.end(); ++it) {
        auto itFeatureVectors = it->second;
        float min = 1000000;
        for (auto itClassifiedFeatureVector = itFeatureVectors.begin(); itClassifiedFeatureVector != itFeatureVectors.end(); ++itClassifiedFeatureVector) {
            auto classifiedFeatures = *itClassifiedFeatureVector;

            if ((classifiedFeatures.size() != featureVector.size())) {
                throw std::logic_error("Classified feature vector size does not equal to input feature vector size!");
            }

            float dotProduct = 0.f, classifiedFeaturesLength = 0.f, featuresLength = 0.f;

            for (auto i = 0; i  < featureVector.size(); ++i) {
                dotProduct += classifiedFeatures[i] * featureVector[i];
                classifiedFeaturesLength += classifiedFeatures[i] * classifiedFeatures[i];
                featuresLength += featureVector[i] * featureVector[i];
            }

            classifiedFeaturesLength = sqrtf(classifiedFeaturesLength);            
            featuresLength = sqrtf(featuresLength);

            float angleCos = dotProduct / (classifiedFeaturesLength * featuresLength);

            if (angleCos < -1.0) angleCos = -1.0 ;
            else if (angleCos > 1.0) angleCos = 1.0 ;

            float angle = acosf(angleCos);

            if (angle < min) {
                min = angle;
            }

            //slog::info << it->first << " angleCos : " << angleCos << slog::endl;
            //slog::info << it->first << " angle : " << angle << slog::endl;
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
