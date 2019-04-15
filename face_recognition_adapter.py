import argparse
import numpy as np
import ctypes as C
import cv2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the image for Face Recognition')

    return parser

libmog = C.cdll.LoadLibrary('face_recognition_prototype.so')

def getfg(img):
    (rows, cols) = (img.shape[0], img.shape[1])
    res = np.zeros(dtype=np.uint8, shape=(rows, cols))
    libmog.getfg(img.shape[0], img.shape[1],
                       img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                       res.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return res


if __name__ == '__main__':
    image_path = get_parser().parse_args().path   
    image = cv2.imread(image_path)
    
    cv2.imshow('Source image', image)
