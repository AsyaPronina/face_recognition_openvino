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

def get_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	return iou

def get_top_one_error(pred_bbox,pred_class,gt_bbox, gt_class, countTrue, countFalse, countAllFaces, countDetectedFaces, countAllImg, countErrImg):
    countFalse+=len(gt_class)
    countAllFaces+=len(gt_bbox)
    countDetectedFaces+=len(pred_bbox)
    countAllImg+=1
    if (len(pred_bbox)!=len(gt_bbox)):
        countErrImg+=1
    for i in range(len(gt_bbox)):
        maxIOU=0
        t=False
        for j in range(len(pred_bbox)):
            IOU=get_iou(pred_bbox[j], gt_bbox[i])
            if (IOU>maxIOU):
                maxIOU=IOU
                if (j>=len(gt_bbox)):
                    t=False
                else:
                    t=pred_class[j]==gt_class[i]
        if maxIOU>=0.5 and t==True:
            countTrue+=1
        if maxIOU<0.5:
            countFalse-=1
    return countTrue, countFalse, countAllFaces, countDetectedFaces, countAllImg, countErrImg

countTrue=0
countFalse=0
countAllFaces=0
countDetectedFaces=0
countAllImg=0
countErrImg=0

path_to_dataset= 'dataset/'
name_dir=["Asyok", "daryafret", "Nastya", "Malinka", "All"]
#mame_dir=["Ion"]
mode="test"

path_predicted = current_path+'/mAP-master/data/predicted/'
path_to_save_for_mAP = current_path+"/mAP-master/input/ground-truth/"
path_to_save_for_other = current_path+'/mAP-master/data/groundtruth/'
path_predicted_for_mAP= current_path+'/mAP-master/input/detection-results/'

#delete files in mAP-master folder in case if tou want run script for not all directories, f.e. only for 'All'
paths_files=[path_predicted, path_to_save_for_other, path_to_save_for_mAP, path_predicted_for_mAP]
for folder in paths_files:
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


#run pipeline
for name in name_dir:
    path_images_in_dir=[]
    p=path_to_dataset+name+"/"+mode+"/";
    for r, d, f in os.walk(p):
        for file in f:
            if '.jpg' in file:
                path_images_in_dir.append(os.path.join(r, file))         

    for image_path in path_images_in_dir:
        image = cv2.imread(image_path)
        fname=os.path.basename(image_path)
        fnameWithoutExt=os.path.splitext(fname)[0]

        path_for_calculate_map=current_path+"/mAP-master/input/detection-results/"+fnameWithoutExt+".txt"
        path_for_result_detection_net=current_path+"/mAP-master/data/predicted/"+fnameWithoutExt+".txt"
        print(image_path)
        detects, recogns, aligns, time  = adapter.recognize_faces(image,path_for_calculate_map,path_for_result_detection_net, None);

#write ground-truth files
for name in name_dir:
    path_to_json_file=path_to_dataset+name+'/'+mode+'/'+name+'_'+mode+'.json'
    print(path_to_json_file)
    if (name!="Ion" or (name=="Ion" and mode=="train")):
        with open(path_to_json_file) as json_file:  
            data = json.load(json_file)
        keys=list(data.keys())
        key = keys[0]
        for key in keys:
            fname = re.search(r'^([^.]+)', data[key]['filename']).group(0)
            final_fname_map=path_to_save_for_mAP+fname+'.txt'
            f_map = open(final_fname_map,"w")
            final_fname_other=path_to_save_for_other+fname+'.txt'
            f_other = open(final_fname_other,"w")
            regions = data[key]['regions']
            for region in regions:
                bbox = region['shape_attributes']
                f_map.write("detection "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')
                f_other.write(region['region_attributes']['class']+" "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')
            f_map.close()
            f_other.close()
    else:
        with open(path_to_json_file) as json_file:  
            data = json.load(json_file)
        keys=list(data.keys())
        key_for_images=keys[1]
        data_image=list(data[key_for_images].keys());
        for key in data_image:
            fname = re.search(r'^([^.]+)', data[key_for_images][key]['filename']).group(0)
            final_fname_map=path_to_save_for_mAP+fname+'.txt'
            f_map = open(final_fname_map,"w")
            final_fname_other=path_to_save_for_other+fname+'.txt'
            f_other = open(final_fname_other,"w")
            regions = data[key_for_images][key]['regions']
            for region in regions:
                bbox = region['shape_attributes']
                f_map.write("detection "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height']) +'\n')
                f_other.write(region['region_attributes']['class']+" "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')    
            f_map.close()
            f_other.close()

#some calculation for top-1
print(path_predicted)
predicred_files = []
for r, d, f in os.walk(path_predicted):
    for file in f:
        predicred_files.append(os.path.join(r, file))

print('predicred: ' + str(predicred_files))
for f in predicred_files:
    with open(f) as file:
        fname=os.path.basename(f)
        fnameWithoutExt=os.path.splitext(fname)[0]
        file_contents = file.read()

    #get predicted data from files
    pred_bbox = []
    pred_class=[]
    allClasses = re.findall(r'(Ion[ \d\.]+|Asyok[ \d\.]+|daryafret[ \d\.]+|Malinka[ \d\.]+|Nastya[ \d\.]+|Unknown[ \d\.]+)', file_contents)
    for c in allClasses:
        rect  = re.findall(r'\d.*', c)[0]
        rect = rect.split()
        pred_bbox.append([int(rect[1]), int(rect[2]), int(rect[3]), int(rect[4])])
        
        c = re.findall(r'[a-zA-z]+', c)
        pred_class.append(c)

    #get groundTruth data from files
    path_gt = path_to_save_for_other+fnameWithoutExt+'.txt'
    with open(path_gt) as gt_file:
        gt_file_contents = gt_file.read()

    gt_bbox = []
    gt_class = []
    gt_allClasses = re.findall(r'(Ion[ \d\.]+|Asyok[ \d\.]+|daryafret[ \d\.]+|Malinka[ \d\.]+|Nastya[ \d\.]+|Unknown[ \d\.]+)', gt_file_contents)
    for c in gt_allClasses:
        gt_rect = re.findall(r'\d.*', c)[0]
        gt_rect = gt_rect.split()
        gt_bbox.append([int(gt_rect[0]), int(gt_rect[1]), int(gt_rect[2]), int(gt_rect[3])])

        c = re.findall(r'[a-zA-z]+', c)
        gt_class.append(c)

    countTrue,countFalse,countAllFaces,countDetectedFaces,countAllImg,countErrImg =get_top_one_error(pred_bbox,pred_class,gt_bbox, gt_class, countTrue, countFalse, countAllFaces, countDetectedFaces, countAllImg, countErrImg) 


countTrue=countTrue+0.0
#print(countTrue)
#print(countFalse)
print("top-1 error = " + str(countTrue/countFalse*100)+"%")  
countDetectedFaces=countDetectedFaces+0.0
#print("accuracy of detection algorithm = " + str(countDetectedFaces/countAllFaces))
print("accuracy of detection algorithm = " + str(countAllFaces/countDetectedFaces*100)+"%")
countErrImg=countErrImg+0.0
print("detection error rate = " + str(countErrImg/countAllImg*100)+"%")   
   
#subprocess.call("python3 " + os.path.dirname(os.path.abspath(sys.argv[0])) + "/main.py", shell=True)
subprocess.call("python3 " + os.path.dirname(os.path.abspath(sys.argv[0])) + "/mAP-master/main.py", shell=True)


