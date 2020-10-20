from PIL import Image
import numpy
import numpy as np

# origin image, processed image
def score(origin_image, out_image):
    return 20*np.log(np.linalg.norm(origin_image, ord=2) / np.linalg.norm(origin_image-out_image, ord=2))

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)
# read the image , change path to process different images
src = numpy.array(Image.open("/home/wang_shuai/图片/1.jpeg").convert("L").resize((200,200)))

src_guass_noise = src.copy() + numpy.random.randn(src.size).reshape(src.shape) * 10

salt0  = numpy.random.random(src.size).reshape(src.shape) < 0.05
salt255  = numpy.random.random(src.size).reshape(src.shape) < 0.05
src_salt_noise = src.copy()
src_salt_noise[salt0] = 0
src_salt_noise[salt255] = 255

gauss_denoise = numpy.asarray(src, dtype=numpy.float)
mean_denoise = numpy.zeros_like(src, dtype=numpy.float)
sobel = numpy.zeros_like(src, dtype=numpy.float)

#高斯kernel （size， \sigma）
gauss_kernel = gkern(3,10)
sobel_kernel = numpy.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)


#filter process
for i in range(1,src.shape[0] - 1):
    for j in range(1,src.shape[1] - 1):
        gauss_denoise[i,j] = (gauss_kernel * src_guass_noise[i-1:i+2, j-1:j+2]).sum() # 高斯滤波
        mean_denoise[i,j] = numpy.sort(src_salt_noise[i-1:i+2, j-1:j+2].reshape(-1))[4] # 均值滤波
        sobel[i,j] = (sobel_kernel * src[i-1:i+2, j-1:j+2]).sum()    # sobel x


gauss_denoise_show = Image.fromarray(gauss_denoise)
gauss_denoise_show.show("高斯滤波效果")
mean_denoise_show = Image.fromarray(mean_denoise)
mean_denoise_show.show("均值滤波效果")
src_salt_noise_show = Image.fromarray(src_salt_noise)
src_salt_noise_show.show("椒盐噪声")
src_guass_noise_show = Image.fromarray(src_guass_noise)
src_guass_noise_show.show("高斯噪声")
sobel_show = Image.fromarray(sobel)

#SNR 信噪比,
print(score(src_guass_noise, gauss_denoise))
sobel_show.show("sobel")

