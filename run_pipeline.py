#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import re
import json
import sys,os
import subprocess
import cv2

current_path=os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, current_path)

import face_recognition_adapter as adapter

path_to_dataset= os.path.join(r'C:\Study\DL\repository\PhotosForFaceRecognition\dataset')
#name_dir=["Asyok", "Ion", "Nastya", "daryafret", "Malinka"] #"Unknown" is for next launch
#"All" was temporarily disabled
name_dir=["Unknown"]
#mode=["train" , "test"]
mode="train"

for name in name_dir:
    path_images_in_dir=[]
    p=os.path.join(os.path.join(path_to_dataset, name), mode);
    print(p)
    for r, d, f in os.walk(p):
        for file in f:
            if '.jpg' in file:
                path_images_in_dir.append(os.path.join(r, file))         

    for image_path in path_images_in_dir:
        print(image_path + ' is handled!\n')
        image = cv2.imread(image_path)
        fname=os.path.basename(image_path)
        fnameWithoutExt=os.path.splitext(fname)[0]

        path_for_calculate_map=os.path.join(os.path.join(current_path, r'\mAP-master\input\detection-results'), (fnameWithoutExt+'.txt'))
        path_for_result_detection_net=os.path.join(os.path.join(current_path+r'\mAP-master\data\predicted'), (fnameWithoutExt+'txt'))
        
        adapter.set_class_for_image(name)
        detects, recogns, aligns, time  = adapter.recognize_faces(image,path_for_calculate_map,path_for_result_detection_net)
        adapter.clear_class()
        
adapter.dump_feature_vectors_to_json("train_data_unknown.json")