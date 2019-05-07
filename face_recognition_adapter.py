#!/usr/bin/python3

import argparse
import numpy as np
import ctypes as C
import cv2
import os
import sys

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the image for Face Recognition')

    return parser

current_path=os.path.dirname(os.path.abspath(sys.argv[0]))
path_library=current_path+'/libface_recognition.so'
#face_recognition = C.cdll.LoadLibrary('/home/anastasiia/Documents/project/web/libface_recognition.so')
face_recognition = C.cdll.LoadLibrary(path_library)

def recognize_faces(image, path_for_calculate_map, path_for_result_detection_net):
    (rows, cols, depth) = (image.shape[0], image.shape[1], image.shape[2])
    detection_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    recognition_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    path_map = path_for_calculate_map.encode('utf-8') 
    path_res = path_for_result_detection_net.encode('utf-8')

    face_recognition.recognizeFaces(image.ctypes.data_as(C.POINTER(C.c_ubyte)), rows, cols,
                                    detection_results.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                    recognition_results.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                    C.c_char_p(path_map),
                                    C.c_char_p(path_res)
                                    )

    aligned_faces_count = face_recognition.getAlignedFacesCount()
    align_width = np.zeros(dtype=np.uint32, shape=(1, aligned_faces_count))
    align_height = np.zeros(dtype=np.uint32, shape=(1, aligned_faces_count))

    face_recognition.getAlignedFacesSizes(align_width.ctypes.data_as(C.POINTER(C.c_uint)),
                                          align_height.ctypes.data_as(C.POINTER(C.c_uint)))


    align_cols = 0
    align_rows = 0

    for i in range(aligned_faces_count):
        align_cols += align_width[0][i]
        align_rows += align_height[0][i]

    align_data = np.zeros(dtype=np.uint8, shape=(1, align_rows * align_cols * depth))
    face_recognition.getAlignedFaces(align_data.ctypes.data_as(C.POINTER(C.c_ubyte)))

    align_results = []
    for i in range(aligned_faces_count):
        width = align_width[0][i]
        height = align_height[0][i]
        size = width * height * depth
      
        face = align_data[:, :size]
        align_data = align_data[:, size:]
        
        align_results.append(face.reshape(height, width, depth))
    
    face_recognition.getFaceRecognitionTime.restype = C.c_double
    recognition_time = face_recognition.getFaceRecognitionTime()
    
    face_recognition.clear()

    return detection_results, recognition_results, align_results, recognition_time

if __name__ == '__main__':
    image_path = get_parser().parse_args().path   
    image = cv2.imread(image_path)

    fname=os.path.basename(image_path)
    fnameWithoutExt=os.path.splitext(fname)[0]
    
    path_for_calculate_map=current_path+"/mAP-master/input/detection-results/"+fnameWithoutExt+".txt"
    path_for_result_detection_net=current_path+"/mAP-master/data/predicted/"+fnameWithoutExt+".txt"
    
    detects, recogns, aligns, time  = recognize_faces(image,path_for_calculate_map,path_for_result_detection_net);

    cv2.imshow('Source image', image)
    cv2.waitKey()
    cv2.imshow('Detected faces', detects)
    cv2.waitKey()
    cv2.imshow('Recognized faces', recogns)
    cv2.waitKey()

    for aligned_image in aligns:
        cv2.imshow('Aligned image', aligned_image)
        cv2.waitKey()

    print("Total time in ms: " + str(time))