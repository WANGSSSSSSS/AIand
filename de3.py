import numpy
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def my_Gauss_filter(src, size):
    h,w = src.shape
    ch,cw = size
    temp = src.copy()

    result = np.zeros_like(src)
    kernel = gkern(3,10)
    print(kernel)
    for i in range(1,h-1):
        for j in range(1,w-1):
            #print(i,j)
            result[i, j] =  (temp[i-1:ch + i-1, j-1:j + cw-1] * kernel).sum()
    return result
def add_noise_salt(src, pred):
    src = src.copy()
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if np.random.random(1) < pred :
                src[i,j]  =  255 if np.random.random(1) > 0.5 else 0
    return src
def add_noise_gauss(src):
    src = src.copy()
    noise = np.random.randn(src.size).reshape(src.shape) * 10
    src = src + noise
    return src
def my_me_filter(src,size):
    h, w = src.shape
    ch, cw = size
    temp = src.copy()

    result = np.zeros_like(src)
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i, j] =  np.sort((temp[i-1:ch + i-1, j-1:j + cw-1]).reshape(-1))[size[0]*size[1] // 2]
    return result

def my_Sobel_filter(src, size):
    h,w = src.shape
    ch,cw = size
    temp  = src.copy()

    result = np.zeros_like(src)
    kernel = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(size)
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i, j] = ( np.abs((temp[i-1:ch + i-1, j-1:j + cw-1] * kernel).sum()))
    return result

def SNR(origin_image, out_image):
    return 20*np.log(np.linalg.norm(origin_image, ord=2) \
                     / np.linalg.norm(origin_image-out_image, ord=2))


image = cv2.imread("/home/wang_shuai/图片/1.jpeg")

image = cv2.resize(image, (512,512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gs_noise = add_noise_gauss(image,)
st_noise = add_noise_salt(image, 0.1)

gs_denoise = my_Gauss_filter(gs_noise, [3,3])
st_denoise = my_me_filter(st_noise, [3,3])


cv_gs = cv2.GaussianBlur(gs_noise, (3,3),10)
cv_st = cv2.medianBlur(st_noise, 3)

print(SNR(gs_noise, gs_denoise))
print(SNR(st_noise, st_denoise))

sobel = my_Sobel_filter(image, [3,3])

cv_sobelx = cv2.Sobel(image,cv2.CV_8U,0, 1, ksize=3)

cv2.imshow("src.jpg", image)
cv2.imshow("gs_noise.jpg", gs_noise / 255)
cv2.imshow("gs_denoise.jpg", gs_denoise / 255)
cv2.imshow("st_noise.jpg", st_noise)
cv2.imshow("st_denoise.jpg", st_denoise)
cv2.imshow("cv_degs.jpg", cv_gs / 255)
cv2.imshow("cv_st.jpg", cv_st)
cv2.imshow("sobel.jpg", sobel)
cv2.imshow("cv_sobel_x.jpg", cv_sobelx)

cv2.waitKey(0)



