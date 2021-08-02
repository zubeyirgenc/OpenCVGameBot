import numpy as np
import pyautogui
import cv2
import datetime
from scipy import signal
import scipy.linalg as la
import scipy
import math
import canny
import scipy.stats as st

def GaussianFilter(sigma, filter_size):
    gaussian_filter = np.zeros((filter_size, filter_size))#, np.float32)
    y_axes = x_axes = np.arange(filter_size)

    xy = np.mgrid[:filter_size,:filter_size]
    power = -(xy[0]**2+xy[1]**2)/(2*(sigma**2))
    gaussian_filter = np.exp(power)

    return gaussian_filter

def take_derivative(J):
    x,y = J.shape
    der = np.zeros((x,y,2))
    der[1:-1,:,0] = (J[2:,:] - J[:-2,:])/2
    der[:,1:-1,1] = (J[:,2:] - J[:,:-2])/2

    return der

def G_x_y(image,filter_size,I):
    Ixx = I[:,:,0]**2
    Iyy = I[:,:,1]**2
    Ixy = I[:,:,0]*I[:,:,1]
    eigens = np.zeros((image.shape[0],image.shape[1],2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Ixx_sum = Ixx[i:i+filter_size, j:j+filter_size].sum()
            Ixy_sum = Ixy[i:i+filter_size, j:j+filter_size].sum()
            Iyy_sum = Iyy[i:i+filter_size, j:j+filter_size].sum()
            
            G = np.array([[Ixx_sum,Ixy_sum], [Ixy_sum,Iyy_sum]])

            w, v = np.linalg.eig(G)
            eigens[i,j,0] = w[0]
            eigens[i,j,1] = w[1]

    return eigens

def corner_finder(image_src,eigen_vec,k = 0.35,line=False,point=True,pixel=False):
    image = np.copy(image_src)
    det = eigen_vec[:,:,0]*eigen_vec[:,:,1]
    trace = eigen_vec[:,:,0]+eigen_vec[:,:,1]
    
    R = det-k*(trace**2)

    if pixel:
        R_great_zero = R>0
        R_great_zero_co = np.nonzero(R_great_zero)
        print(R_great_zero_co[0].shape)
        image[R_great_zero_co[0],R_great_zero_co[1],0]=0
        image[R_great_zero_co[0],R_great_zero_co[1],1]=255    
        image[R_great_zero_co[0],R_great_zero_co[1],2]=0

    if line:
        R_small_zero = R<0
        R_small_zero_co = np.nonzero(R_small_zero)
        print(R_small_zero_co[0].shape)
        image[R_small_zero_co[0],R_small_zero_co[1],0]=0
        image[R_small_zero_co[0],R_small_zero_co[1],1]=0    
        image[R_small_zero_co[0],R_small_zero_co[1],2]=255

    if point:
        R_great_zero_ = R>0
        R_great_zero_co_ = np.nonzero(R_great_zero_)
        for i in range(len(R_great_zero_co_)):
            image = cv2.line(image, (R_great_zero_co_[0][i],R_great_zero_co_[1][i]), (R_great_zero_co_[0][i],R_great_zero_co_[1][i]), (0,255,0), 1)
    return image

def main(is_four,bw=False):
    canny_image,image = canny.canny(is_four)

    cv2.imshow("canny_image", canny_image)

    filter_size = 3
    sigma = math.sqrt(filter_size)   

    gaus_filter = GaussianFilter(sigma, filter_size)

    J = scipy.signal.convolve(canny_image, gaus_filter, mode='same')   
    J[:,:] /= filter_size * filter_size 

    cv2.imshow("J", J)
    print(J.shape)
    I = take_derivative(J)

    eigen_vec = G_x_y(image,3,I)#.astype(np.uint8))
    # eigen_vec = cv2.cornerEigenValsAndVecs(J.astype(np.uint8), 3, 3)[:,:2]
    # print(eigen_vec[0])
    # print(eigen[0])

    if bw:
        corner_bw(image_src,eigen_vec,k = 0.06,line=True,point=False,pixel=False)
    else:
        image = corner_finder(image,eigen_vec,k=0.4,line=False,point=False,pixel=True)
    return image

def corner_bw(is_four):
    canny_image,image = canny.canny(is_four)

    cv2.imshow("canny_image", canny_image)

    filter_size = 3
    sigma = math.sqrt(filter_size)   

    gaus_filter = GaussianFilter(sigma, filter_size)
    J = scipy.signal.convolve(canny_image, gaus_filter, mode='same')
    I = take_derivative(J)

    a = G_x_y(image,3,I)
    # a = cv2.cornerEigenValsAndVecs(J.astype(np.uint8), 3, 3)

    Mc = np.empty(a.shape, dtype=np.float32)
    T = 0.02
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            lambda_1 = a[i, j, 0]
            lambda_2 = a[i, j, 1]
            Mc[i, j] = lambda_1*lambda_2 if lambda_1 * lambda_2 > T else 0 #- 0.04*pow((lambda_1 + lambda_2), 2)
    k=0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    (rows, cols, _) = image.shape
    for row in range(rows):
        for col in range(cols):
            if Mc[row, col]:
                k+=1
                image[row-1:row+2, col-1:col+2] = (0, 255, 0)

    print(k)
    # cv2.imwrite("test_out3.png", image)
    # cv2.imshow("out3", image)
    # cv2.waitKey(0)
    return image

if __name__ == "__main__":
    # image = main(is_four=True)
    image = corner_bw(is_four=True)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    