import cv2,argparse
import os

def parse_args():
    parser = argparse.ArgumentParser('jpeg compression')
    parser.add_argument('img_path',help="Image path")
    args,_ = parser.parse_known_args()
    return args

def jpeg_analysis(img_path,quality=[100]):
    img = cv2.imread(img_path)
    img_basename = os.path.splitext(img_path)[0]
    img_basename = os.path.split(img_path)[1]
    for q in quality:
        out_path = img_basename+'_'+str(q)+'.jpg'
        cv2.imwrite(out_path,img,[cv2.IMWRITE_JPEG_QUALITY,q])

if __name__ == '__main__':
    args = parse_args()
    img_path = args.img_path
    quality = [100,95,90,85,80,75,70,60,50,40,20,10]
    jpeg_analysis(img_path,quality)
