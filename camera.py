import cv2
import numpy

def perpare2d(path):
    x,y = [],[]
    with open(path, "r") as file:
        x = file.readline().split(",")
        y = file.readline().split(",")
    x = [float(x_) for x_ in x]
    y = [float(y_) for y_ in y]
    x = numpy.array(x, dtype=numpy.float)
    y = numpy.array(y, dtype=numpy.float)
    x += numpy.random.random(x.shape)*3
    y += numpy.random.random(x.shape)*3
    one = numpy.ones_like(x, dtype=numpy.float)

    return numpy.stack([x,y,one],axis=1)

def perpare3d(path):
    x, y, z = [], [], []
    with open(path, "r") as file:
        x = file.readline().split(",")
        y = file.readline().split(",")
        z = file.readline().split(",")
    x = [float(x_) for x_ in x]
    y = [float(y_) for y_ in y]
    z = [float(z_) for z_ in z]
    x = numpy.array(x, dtype=numpy.float)
    y = numpy.array(y, dtype=numpy.float)
    z = numpy.array(z, dtype=numpy.float)
    one = numpy.ones_like(x, dtype=numpy.float)
    return numpy.stack([x, y, z, one], axis=1)

def sloveP(p3d, p2d):
    n,_ = p2d.shape
    A = numpy.zeros([2*n,12])
    for i in range(n):
        A[i*2,0:4] = p3d[i,...]
        A[i*2,8:12] =  - p3d[i,...]*p2d[i,0]
        A[i*2+1,4:8] = p3d[i,...]
        A[i*2+1,8:12] =  - p3d[i,...] * p2d[i,1]
    ATA = numpy.matmul(numpy.transpose(A), A)
    eig, eigx = numpy.linalg.eig(ATA)

    eigx = eigx[:, eig.argmin()]
    eigx = eigx / eigx[-1]
    return eigx.transpose()

def computeP(P):
    K,R = numpy.linalg.qr(P)
    return K,R

if __name__ == "__main__":
    p2d = perpare2d("point2d.txt")
    p3d = perpare3d("point3d.txt")
    im  = cv2.imread("/home/wang_shuai/Assignment-4-Material/stereo2012b.jpg")
    # for p in p2d:
    #     print(p[:2])
    #     cv2.circle(im, (int(p[0]),int(p[1])), 5,(100,200,100), 2)

    #cv2.imshow("im", im)
    cv2.waitKey()
    P = sloveP(p3d, p2d)
    print("P:")
    print(P)
    c = cv2.decomposeProjectionMatrix(P.reshape(3,4))
    print("K:")
    print(c[0] / c[0][2,2])

    print("R:")
    print(c[1])

    print("T:")
    print(c[2] / c[2][-1])


 # b = p2d.reshape(-1,1)
    # AT = numpy.transpose(A)
    # ATA = numpy.matmul(AT,A)
    # ATb = numpy.matmul(AT,b)
    # ATA_ = numpy.linalg.inv(ATA)
    # return numpy.matmul(ATA_,ATb)