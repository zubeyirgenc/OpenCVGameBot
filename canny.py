import cv2
import numpy as np
import sobel
import datetime

def canny(is_four=False):
    edge,img = sobel.sobel(is_four)

    canny_ = cv2.Canny(edge,500,350)#500,350
    return canny_,img

if __name__ == "__main__":
    a = datetime.datetime.now()
    edge,img = canny()
    b = datetime.datetime.now()
    print("1280x720 pixel images Take Screenshot + Sobel + Canny time is ",b-a)
    edge = cv2.resize(edge, dsize=(int(edge.shape[1]/2),int(edge.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("image", edge)
    cv2.waitKey(0)