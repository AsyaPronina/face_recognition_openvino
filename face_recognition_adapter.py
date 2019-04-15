import argparse
import numpy as np
import ctypes as C
import cv2
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the image for Face Recognition')

    return parser

face_recognition = C.cdll.LoadLibrary('libface_recognition.so')

def recognizeFaces(image):
    (rows, cols, depth) = (image.shape[0], image.shape[1], image.shape[2])
    detection_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    recognition_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    face_recognition.recognizeFaces(image.ctypes.data_as(C.POINTER(C.c_ubyte)), rows, cols,
                                    detection_results.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                    recognition_results.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return detection_results, recognition_results

if __name__ == '__main__':
    image_path = get_parser().parse_args().path   
    image = cv2.imread(image_path)
    
    detects, recogns = recognizeFaces(image);
    cv2.imshow('Source image', image)
    cv2.imshow('Detected faces', detects)
    cv2.imshow('Recognized faces', recogns)
    cv2.waitKey()
