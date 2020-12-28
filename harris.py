import numpy
import cv2


def get_derive(image):
    dx = numpy.asarray(image, dtype=numpy.float)
    dy = numpy.asarray(image, dtype=numpy.float)
    _dx = numpy.array([-1,-1,-1, 0,0,0,1,1,1], dtype=numpy.float).reshape([3,3])
    _dy = _dx.copy().transpose(1,0)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            dx[i,j] = (image[i-1:i+2, j-1:j+2] * _dx).sum()
            dy[i,j] = (image[i-1:i+2, j-1:j+2] * _dy).sum()
    return dx, dy
def sum_derive(derive, size = [3,3]):
    sum_ = numpy.zeros(derive.shape, dtype=numpy.float)
    for i in range(1,derive.shape[0]-1):
        for j in range(1,derive.shape[1]-1):
            sum_[i,j] = (derive[i - 1:i + 2,j - 1:j + 2]).sum()
    return sum_

def compute(image):
    dx,dy = get_derive(image)
    dx2 =sum_derive(dx*dx)
    dxdy = sum_derive(dx*dy)
    dy2 = sum_derive(dy*dy)
    H = numpy.zeros((*image.shape, 2,2), dtype=numpy.float)
    H[:,:,0,0] = dx2
    H[:,:,0,1] = dxdy
    H[:,:,1,0] = dxdy
    H[:,:,1,1] = dy2
    R = numpy.zeros(image.shape, dtype=numpy.float)
    for i in range(1,H.shape[0]-1):
        for j in range(1,H.shape[1]-1):
             #print(H[i,j].shape)
             R[i,j]= numpy.linalg.det(H[i,j]) - 0.04*(numpy.ma.trace(H[i,j]))**2
    return R
def nms(src, threshold=1):
    m = numpy.zeros(src.shape, dtype=numpy.float)
    # print(src.max(), src.min())
    for i in range(1,src.shape[0]-1):
        for j in range(1,src.shape[1]-1):
            m[i, j] = (src[i - 1:i + 2, j - 1:j + 2]).max()
    keep = (m == src) & (m > 0.01*m.max())

    tem = numpy.zeros(keep.shape)
    if threshold == 1:
        for i in range(1, src.shape[0] - 1):
            for j in range(1, src.shape[1] - 1):
               if keep[i,j] >0 :
                    cv2.circle(tem, (j,i), 4,1)
    return tem > 0

if __name__ =="__main__" :
    image = cv2.imread("/home/wang_shuai/图片/1.jpeg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corner = cv2.cornerHarris(gray,3,3,0.04)

    F = nms(corner)
    cv_image = image.copy()
    cv_image[F] = [0, 0, 255]
    cv2.imshow("cv_harris", cv_image)
    cv2.waitKey()


    # cv2.imshow("image", image)
    # cv2.waitKey(100)
    # R = compute(gray)
    # F = nms(R)
    # image[F] = [0, 255, 0]
    # cv2.imshow("harris", image)
    # cv2.waitKey()