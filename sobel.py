import cv2
import pyautogui
import numpy as np
from scipy import signal
import datetime

def take_screen(is_four=False):
    myScreenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(myScreenshot), cv2.COLOR_RGB2BGR) 
    if is_four:   
        img = img[580:,505:,:]
        img = cv2.resize(img, dsize=(500, 100), interpolation=cv2.INTER_CUBIC)
    return img

def sobel(is_four=False):    
    img = take_screen(is_four)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter_v = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    filter_h = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1,-2,-1]])

    sobel_x = signal.convolve(gray, filter_h)
    sobel_y = signal.convolve(gray, filter_v)

    euc_dist = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    euc_dist *= 255.0 / euc_dist.max()    
    euc_dist = euc_dist.astype(np.uint8)

    return euc_dist, img

def opencv_sobel():
    img = take_screen(False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
if __name__ == "__main__":
    a = datetime.datetime.now()
    edge,img = sobel()
    b = datetime.datetime.now()
    print("1280x720 pixel images Take Screenshot + Sobel time is ",b-a)    
    c = datetime.datetime.now()
    opencv_sobel()    
    d = datetime.datetime.now()
    print("1280x720 pixel images Take Screenshot + opencv_sobel time is ",d-c)
    edge = cv2.resize(edge, dsize=(int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("screen_sobel",edge)
    cv2.waitKey(0)