import cv2
import numpy as np

def gaussian_kernel_2d_opencv(kernel_size = [3,3],sigma = 10):
    kx = cv2.getGaussianKernel(kernel_size[1],sigma)
    ky = cv2.getGaussianKernel(kernel_size[0],sigma)
    return np.multiply(kx,np.transpose(ky))
def my_Gauss_filter(src, size, stride, padding):
    h,w = src.shape
    ch,cw = size
    temp  = np.zeros([h+padding*2,w+padding*2], dtype=np.float)
    temp[padding:padding+h, padding:padding+w] = src.copy()

    result = np.zeros_like(src)
    kernel = gaussian_kernel_2d_opencv(size)
    print(kernel)
    for i in range(h//stride):
        for j in range(w//stride):
            #print(i,j)
            result[i, j] =  (temp[i*stride:ch+i*stride, j*stride:j*stride+cw] * kernel).sum()
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
def my_me_filter(src,size, stride, padding):
    h, w = src.shape
    ch, cw = size
    temp = np.zeros([h + padding * 2, w + padding * 2], dtype=np.float)
    temp[padding:padding + h, padding:padding + w] = src.copy()

    result = np.zeros_like(src)
    for i in range(h // stride):
        for j in range(w // stride):
            result[i, j] =  np.sort((temp[i*stride:ch+i*stride, j*stride:j*stride+cw]).reshape(-1))[size[0]*size[1] // 2]
    return result

def my_Sobel_filter(src, size, stride, padding):
    h,w = src.shape
    ch,cw = size
    temp  = np.zeros([h+padding*2,w+padding*2], dtype=np.float)
    temp[padding:padding + h, padding:padding + w] = src.copy().astype(np.float)

    result = np.zeros_like(src)
    kernel = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(size)
    for i in range(h // stride):
        for j in range(w // stride):
            result[i, j] = ( np.abs((temp[i * stride:ch + i * stride, j * stride:j * stride + cw] * kernel).sum()))
    result = my_me_filter(result, [3,3], 1,1)
    return result

def score(origin_image, out_image):
    return 20*np.log(np.linalg.norm(origin_image, ord=2) / np.linalg.norm(origin_image-out_image, ord=2))

if __name__ =="__main__":
    image = cv2.imread("/home/wang_shuai/图片/1.jpeg")

    image = cv2.resize(image, (512,512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gs_noise = add_noise_gauss(image,)
    st_noise = add_noise_salt(image, 0.1)

    gs_denoise = my_Gauss_filter(gs_noise, [3,3], 1, 1)
    st_denoise = my_me_filter(st_noise, [3,3], 1,1)


    cv_gs = cv2.GaussianBlur(gs_noise, (3,3),10)
    cv_st = cv2.medianBlur(st_noise, 3)

    print(score(gs_noise, gs_denoise))
    print(score(st_noise, st_denoise))

    sobel = my_Sobel_filter(image.copy(), [3,3], 1, 1)

    cv_sobelx = cv2.Sobel(image,cv2.CV_8U,0, 1, ksize=3)

    cv2.imshow("src.jpg", image)
    cv2.imshow("gs_noise.jpg", gs_noise / 255)
    cv2.imshow("gs_denoise.jpg", gs_denoise /255)
    cv2.imshow("st_noise.jpg", st_noise)
    cv2.imshow("st_denoise.jpg", st_denoise)
    cv2.imshow("cv_degs.jpg", cv_gs/ 255)
    cv2.imshow("cv_st.jpg", cv_st)
    cv2.imshow("sobel.jpg", sobel)
    cv2.imshow("cv_sobel_x.jpg", cv_sobelx)
    #cv2.imshow("cv_sobel_y", cv_sobely)

    cv2.waitKey()