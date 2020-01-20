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

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=MAX_FEATURES)
    #sift = cv2.xfeatures2d_SIFT(nfeatures=MAX_FEATURES)
    kp1,desc1 = sift.detectAndCompute(gray_img1,None)
    kp2,desc2 = sift.detectAndCompute(gray_img2,None)

    #matches.sort(key=lambda x: x.distance, reverse=False)
    #numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    #matches = matches[:numGoodMatches]

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    matches = good
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