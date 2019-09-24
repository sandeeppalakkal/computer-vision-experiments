# Image rotation - naive implementation
# If image should be rotated by theta, image should be resampled by a rotation matrix with -theta
# Rotation matrix = [cos theta -sin theta
#                    sin theta cos theta]
# This is because if we are rotating image by theta, we are equivalently rotating the original coordinate axes by -theta. 
# In camera calibration matrix, if camera is rotated by theta, the coordinate axes are rotated by theta. Hence image gets rotated by -theta. So, if R(theta) is the camera rotation matrix, the image is equivalently rotated by -theta.

import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Image rotation')
    parser.add_argument('img_path',help='Image path')
    parser.add_argument('angle',type=float,help='Angle of rotation in degrees')
    args,_ = parser.parse_known_args()
    return args

def rotated_value(x,y,theta):
    '''Rotation of (x,y) by an angle theta (counter-clockwise)'''
    c = np.cos(theta)
    s = np.sin(theta)
    x_ = x * c - y * s
    y_ = x * s + y * c
    return x_,y_

def bounded(x,l,u):
    if x < u and x >= l:
        return True
    else: return False

def _rotate(img,angle):
    h,w = img.shape[0:2]
    theta = angle * np.pi / 180
    # To rotate image by theta, resample the image by -theta
    x1,y1 = rotated_value(w/2.,-h/2.,-theta)
    x2,y2 = rotated_value(w/2.,h/2.,-theta)
    x3,y3 = rotated_value(-w/2.,h/2.,-theta)
    x4,y4 = rotated_value(-w/2.,-h/2.,-theta)
    w_= np.ceil(max(x1,x2,x3,x4) - min(x1,x2,x3,x4)).astype(int)
    h_ = np.ceil(max(y1,y2,y3,y4) - min(y1,y2,y3,y4)).astype(int)
    img_rot = np.zeros((h_,w_),dtype=np.uint8)

    # Naive loops, slow for large images
    for x in range(w_):
        for y in range(h_):
            x_,y_ = rotated_value(x-w_/2.,y-h_/2.,theta)
            x_ += w/2.
            y_ += h/2.
            xl = np.floor(x_).astype(int)
            xu = np.floor(x_+1).astype(int)
            yl = np.floor(y_).astype(int)
            yu = np.floor(y_+1).astype(int)
            if not (bounded(xl,0,w) and bounded(xu,0,w) and bounded(yl,0,h) and bounded(yu,0,h)):
                continue
            # Bilinear interpolation
            imgl = float(xu - x_) * img[yl,xl] + float(x_ - xl) * img[yl,xu]
            imgu = float(xu - x_) * img[yu,xl] + float(x_ - xl) * img[yu,xu]
            imgc = float(yu - y_) * imgl + float(y_ - yl) * imgu
            img_rot[y,x] = imgc.astype(np.uint8)

    return img_rot

def rotate(img_path,angle):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Could not read the image')
        return
    img_rot = _rotate(img,angle)
    cv2.imshow("Input",img)
    cv2.imshow("Rotated",img_rot)
    cv2.waitKey()
    return

if __name__ == '__main__':
    args = parse_args()
    img_path = args.img_path
    angle = args.angle
    rotate(img_path,angle)