import os,pdb,random
import cv2
from matplotlib import pyplot as plt
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Scale calculation")
    parser.add_argument('img1',help='First image path')
    parser.add_argument('img2',help='Second image path')
    args,_ = parser.parse_known_args()
    return args

def compute_scale_change(img_path1,img_path2):
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.1

    img1 = cv2.imread(img_path1,cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2,cv2.IMREAD_COLOR)

    gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES,1.2)
    kp1,desc1 = orb.detectAndCompute(gray_img1,None)
    kp2,desc2 = orb.detectAndCompute(gray_img2,None)

    #matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    #matcher = cv2.DescriptorMatcher_create("BruteForce-HammingLUT")
    #matcher = cv2.DescriptorMatcher_create("FlannBased")
    #matches = matcher.match(desc1,desc2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(desc1,desc2)

    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(img1,kp1,img2,kp2,matches[:3],None)
    cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Find homography
    # The estimated homography will be stored in h
    hm, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    ids = random.sample(range(0,points1.shape[0]),3)
    #M = cv2.getAffineTransform(points1[ids,:],points2[ids,:])
    
    # Use homography
    # Registered image will be resotred in imReg
    height, width, channels = img2.shape
    im1Reg = cv2.warpPerspective(img1, hm, (width, height))
    
    print("Estimated homography : \n",hm)
    #print("Estimated affine: \n",M)
    #print("Estimated scale : %f" : scale_est)

    cv2.imwrite("img1_aligned.jpg", im1Reg)

if __name__ == '__main__':
    args = parse_args()
    img1 = args.img1
    img2 = args.img2
    compute_scale_change(img1,img2)