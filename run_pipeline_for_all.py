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


def execute():
   mode = 'train'
  
   path_to_json_file='dataset/All/' + mode + '/All_' + mode + '.json'
   print(path_to_json_file)
   with open(path_to_json_file) as json_file:  
       data = json.load(json_file)
   keys=list(data.keys())
   key = keys[0]
   for key in keys:
       ground_truth = {}
       
       fname = re.search(r'^([^.]+)', data[key]['filename']).group(0)
       print(fname + " is handled!")
       regions = data[key]['regions']
       for region in regions:
           ground_truth[region['region_attributes']['class']] = region['shape_attributes']
      
       image_path = 'dataset/All/' + mode + '/' + fname + '.jpg'
       print(image_path)    
       image = cv2.imread(image_path)
       fname=os.path.basename(image_path)
       fnameWithoutExt=os.path.splitext(fname)[0]

       path_for_calculate_map=current_path+"/mAP-master/input/detection-results/"+fnameWithoutExt+".txt"
       path_for_result_detection_net=current_path+"/mAP-master/data/predicted/"+fnameWithoutExt+".txt"
 

       detects, recogns, aligns, time  = adapter.recognize_faces(image,path_for_calculate_map,path_for_result_detection_net, ground_truth)
       cv2.imwrite('recognitions/' + fname + '.jpg', recogns)

if __name__ == "__main__":
    execute()
    adapter.dump_feature_vectors_to_json('train_data_all.json')
