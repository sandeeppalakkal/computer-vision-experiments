# Image rotation - opencv warpAffine function

import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Image rotation')
    parser.add_argument('img_path',help='Image path')
    parser.add_argument('angle',type=float,help='Angle of rotation in degrees')
    args,_ = parser.parse_known_args()
    return args

def rotate(img_path,angle):
    img = cv2.imread(img_path)#,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Could not read the image')
        return
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.,h/2.),angle,1)
    img_rot = cv2.warpAffine(img,M,(w,h))
    cv2.imshow("Input",img)
    cv2.imshow("Rotated",img_rot)
    cv2.waitKey()
    return

if __name__ == '__main__':
    args = parse_args()
    img_path = args.img_path
    angle = args.angle
    rotate(img_path,angle)