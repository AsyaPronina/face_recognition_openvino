import os, sys
import json

train_data = None
with open('train_data.json') as train_data_json:
    train_data = json.load(train_data_json)

with open('train_data_converted.txt', 'w') as train_data_text:
    if train_data != None:
        labels = train_data['labels']

        firstLabel = labels[0]
        feature_vectors = train_data[firstLabel]
        train_data_text.write('{ "' + firstLabel + '", {')

        first_feature_vector = feature_vectors[0]
        train_data_text.write('\n{')
        float_feature_vector = [ float(n) for n in first_feature_vector.split(', ') ]
        train_data_text.write(str(float_feature_vector[0]))
        for i in range(1, len(float_feature_vector)):
            train_data_text.write(',' + str(float_feature_vector[i]))
        train_data_text.write('}')
            
        for i in range(1, len(feature_vectors)):
            feature_vector = feature_vectors[i]
            train_data_text.write(',\n{')
            float_feature_vector = [ float(n) for n in feature_vector.split(', ') ]
            train_data_text.write(str(float_feature_vector[0]))
            for j in range(1, len(float_feature_vector)):
                 train_data_text.write(',' + str(float_feature_vector[j]))
            train_data_text.write('}')

        train_data_text.write('}\n}')

        labels.remove(firstLabel)        
        for label in labels:
            feature_vectors = train_data[label]
            train_data_text.write(',\n{ "' + label + '", {')

            first_feature_vector = feature_vectors[0]
            train_data_text.write('\n{')
            float_feature_vector = [ float(n) for n in first_feature_vector.split(', ') ]
            train_data_text.write(str(float_feature_vector[0]))
            for i in range(1, len(float_feature_vector)):
                train_data_text.write(',' + str(float_feature_vector[i]))
            train_data_text.write('}')
            
            for i in range(1, len(feature_vectors)):
                feature_vector = feature_vectors[i]
                train_data_text.write(',\n{')
                float_feature_vector = [ float(n) for n in feature_vector.split(', ') ]
                train_data_text.write(str(float_feature_vector[0]))
                for j in range(1, len(float_feature_vector)):
                    train_data_text.write(',' + str(float_feature_vector[j]))
                train_data_text.write('}')

            train_data_text.write('}\n}')
    
    
