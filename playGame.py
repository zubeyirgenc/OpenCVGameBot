import numpy as np
import cv2
import harris_corner_detector
import canny
import time
import math
from scipy import signal
import pyautogui
import datetime

def find_countors(canny_image):

    filter_size = 3
    sigma = math.sqrt(filter_size)
    gaus_filter = harris_corner_detector.GaussianFilter(sigma, filter_size)
    J = signal.convolve(canny_image, gaus_filter, mode='same')   
    J[:,:] /= filter_size * filter_size 

    contours , hierarchy = cv2.findContours(J.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def main_loop(canny_image, y_start, y_stop):

    canny_image,img = canny.canny(is_four=True)
    contours = find_countors(canny_image)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.005* cv2.arcLength(contour, True), True)
        approx_ravel = approx

        if len(approx) == 3:
            left_y = approx_ravel[0][0][0]
            right_y = approx_ravel[2][0][0]

            if y_stop-y_start<100:

                if left_y<y_stop and left_y>y_start:
                    pyautogui.keyDown("a")
                    time.sleep(0.1)
                    pyautogui.keyUp("a")
                    print("triangle solu pembede")

                elif right_y<y_stop and right_y>y_start:
                    pyautogui.keyDown("a")
                    time.sleep(0.1)
                    pyautogui.keyUp("a")
                    print("triangle sagi pembede")

        elif len(approx) == 4 :
            left_y = approx_ravel[0][0][0]
            right_y = approx_ravel[2][0][0]

            if y_stop-y_start<100:

                if left_y<y_stop and left_y>y_start:
                    pyautogui.keyDown("s")
                    time.sleep(0.1)
                    pyautogui.keyUp("s")
                    print("rectangle solu pembede")

                elif right_y<y_stop and right_y>y_start:
                    pyautogui.keyDown("s")
                    time.sleep(0.1)
                    pyautogui.keyUp("s")
                    print("rectangle sagi pembede")

        elif len(approx) == 6 :
            left_y = approx_ravel[0][0][0]  
            right_y = approx_ravel[4][0][0]

            if y_stop-y_start<100:

                if left_y<y_stop and left_y>y_start:
                    pyautogui.keyDown("f")
                    time.sleep(0.1)
                    pyautogui.keyUp("f")
                    print("heksagon solu pembede")

                elif right_y<y_stop and right_y>y_start:
                    pyautogui.keyDown("f")
                    time.sleep(0.1)
                    pyautogui.keyUp("f")
                    print("heksagon sagi pembede")

        elif len(approx) == 10:
            left_y = approx_ravel[0][0][0]  
            right_y = approx_ravel[4][0][0]

            if y_stop-y_start<100:

                if left_y<y_stop and left_y>y_start:
                    pyautogui.keyDown("d")
                    time.sleep(0.1)
                    pyautogui.keyUp("d")
                    print("star solu pembede")

                elif right_y<y_stop and right_y>y_start:
                    pyautogui.keyDown("d")
                    time.sleep(0.1)
                    pyautogui.keyUp("d")
                    print("star sagi pembede")

def main():   
    canny_image,img = canny.canny(is_four=True)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        reg = img_gray[0]==210
        nonz = np.nonzero(reg)
        y_start = nonz[0][0]
        y_stop = nonz[0][-1]
    except:
        y_start = 66
        y_stop = 139
    a = 0
    try:
        
        while True:
            a = datetime.datetime.now()
            main_loop(canny_image, y_start, y_stop)
            print("FPS: ",1/(datetime.datetime.now()-a).total_seconds())

    except KeyboardInterrupt:
        print('interrupted!')

if __name__ == "__main__":
    main()