import cv2
import numpy as np
import numpy


image1 = cv2.imread("/home/wang_shuai/Assignment-4-Material/Left.jpg")
image2 = cv2.imread("/home/wang_shuai/Assignment-4-Material/Right.jpg")


# kp1 = np.array([kp1[g.queryIdx].pt for g in good])
# kp2 = np.array([kp2[g.trainIdx].pt for g in good])

def perpare2d(path):
    x,y = [],[]
    with open(path, "r") as file:
        x = file.readline().split(", ")
        y = file.readline().split(", ")
    x = [float(x_) for x_ in x]
    y = [float(y_) for y_ in y]
    x = numpy.array(x, dtype=numpy.float)
    y = numpy.array(y, dtype=numpy.float)

    return numpy.stack([x,y],axis=1)
kp1 = perpare2d("Homopoint2d_Left.txt")
kp2 = perpare2d("Homopoint2d_Right.txt")
# 可视化
h,w,c= image1.shape
img3 = numpy.zeros([h,w*2,c], dtype=np.uint8)
img3[:, 0:w] = image1
img3[:,w:2*w] = image2

for i in range(6):
    x1 = int(kp1[i][0])
    y1 = int(kp1[i][1])
    x2 = int(kp2[i][0])+w
    y2 = int(kp2[i][1])
    cv2.line(img3, (x1,y1), (x2,y2), (100,200,100), 2)



cv2.imshow("match", img3)
cv2.waitKey()
#



def solve(kp1, kp2):
    n,_ = kp1.shape
    A = np.zeros([n*2,9])
    for i in range(n):
        A[2*i,0:2] = kp1[i]
        A[2*i,2] = 1
        A[2*i,6:8] = -kp1[i]*kp2[i,0]
        A[2*i,8] = -kp2[i][0]

        A[2*i+1,3:5] = kp1[i]
        A[2*i+1,5] = 1
        A[2*i+1,6:8] = -kp1[i]*kp2[i,1]
        A[2*i+1,8] = -kp2[i][1]
    ATA = np.matmul(np.transpose(A), A)
    eig, eigx = np.linalg.eig(ATA)
    eigx = eigx[:,eig.argmin()]
    eigx = eigx / eigx[-1]
    return eigx.transpose()


x = solve(kp1, kp2)


x = x.reshape(3,3)
print("using my:")
print(x)
# print(kp2[0])

t =  np.array([kp1[0][0],kp1[0][1],1])
y = np.matmul(x,t)
# print(y / y[2])

homo,_  = cv2.findHomography(kp1,kp2,0)
print("using opencv:")
print(homo)
# y = np.matmul(homo,t)
# print(y/y[-1])