import cv2
import numpy as np

def gaussian_kernel_2d_opencv(kernel_size = [3,3],sigma = 0):
    kx = cv2.getGaussianKernel(kernel_size[1],sigma)
    ky = cv2.getGaussianKernel(kernel_size[0],sigma)
    return np.multiply(kx,np.transpose(ky))
def my_Gauss_filter(src, size, stride, padding):
    h,w = src.shape
    ch,cw = size
    temp  = np.zeros([h+padding*2,w+padding*2], dtype=np.int8)
    kernel = gaussian_kernel_2d_opencv(size)
    for i in range((h+padding*2)//stride):
        for j in range((w+padding*2)//stride):
            temp[i+size[0]//2, j+size[1]//2] =  (src[i:ch+i, j:j+cw] * kernel).sum()
    return temp
def add_noise_salt(src, pred):

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if np.random.random(1) > pred :
                src[i,j]  =  255 if np.random.random(1) > 0.5 else 0
    return src
def add_noise_gauss(src):
    noise = np.random.randn(src.size).reshape(src.shape) * 10
    src = src + noise
    src.astype(np.int8)
    return src
def my_me_filter(src,size, stride, padding):
    h,w = src.shape
    ch,cw = size
    temp  = np.zeros([h+padding*2,w+padding*2], dtype=np.int8)
    kernel = gaussian_kernel_2d_opencv(size)
    for i in range((h+padding*2)//stride):
        for j in range((w+padding*2)//stride):
            temp[i + size[0]//2, j+size[1]//2] =  np.sort((src[i:ch+i, j:j+cw]).reshape(-1))[size[0]*size[1] // 2]
    return temp
def my_Sobel_filter(src, size, stride, padding):
    h,w = src.shape
    ch,cw = size
    temp  = np.zeros([h+padding*2,w+padding*2], dtype=np.int8)
    kernel = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(size)
    for i in range((h+padding*2)//stride):
        for j in range((w+padding*2)//stride):
            temp[i+size[0]//2, j+size[1]//2] =  (src[i:ch+i, j:j+cw] * kernel).sum()
    return temp

if __name__ =="__main__":
    image = cv2.imread("/home/wang_shuai/图片/1.jpg")