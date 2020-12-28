import cv2
import numpy


Img = cv2.imread("/home/wang_shuai/test2/1073445.jpg")
Img = cv2.resize(Img, (2000,1200))

num = 0

for i in range(0,2000,200):
    for j in range(0, 1200, 200):
        img = Img[j:j+200, i:i+200]
        cv2.imshow("img", img)
        cv2.waitKey(100)
        cv2.imwrite("/home/wang_shuai/test2/sub_{}.jpg".format(num), img)
        num +=1
it = 0
num_ = [0]*60
Img_ = numpy.zeros((2000,1200,3), numpy.float)
for i in range(0,1200,200):
    for j in range(0, 2000, 200):
        it = numpy.random.randint(0,60)
        while num_[it] == 1 :
            it = numpy.random.randint(0, 60)
        num_[it] = 1
        Img_[j:j + 200, i:i + 200] = cv2.imread("/home/wang_shuai/test2/sub_{}.jpg".format(it))/255
        it +=1
        print("{} {}".format(i,j))
Img_ = cv2.resize(Img_,(600,1000))
cv2.imshow("img", Img_)
cv2.waitKey()

