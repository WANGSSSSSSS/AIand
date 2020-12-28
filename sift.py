import numpy

import cv2

def rotate(image, rot=0, ratio=1.0):
    h,w,c = image.shape
    M = cv2.getRotationMatrix2D((w/2,h/2),rot, ratio)
    result = cv2.warpAffine(image,M,(int(w),int(h)))
    return result

def match(image1, image2, des=None):
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)
    output = cv2.drawKeypoints(image1, kp1, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good,
                              None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(des, img3)
    cv2.waitKey(100)

if __name__ == "__main__":
    image = cv2.imread("/home/wang_shuai/图片/1.jpeg")

    image_30 = rotate(image, 30,1)
    image_90 = rotate(image, 90, 1)
    image_180 = rotate(image, 180,1)
    image_0_8 = rotate(image, 0, 0.8)
    image_0_12 = rotate(image, 0, 1.2)
    image_0_14 = rotate(image, 0, 1.4)

    match(image, image_30, "30")
    match(image, image_90, "90")
    match(image, image_180, "180")
    match(image, image_0_8, "0.8")
    match(image, image_0_12, "1.2")
    match(image, image_0_14, "1.4")

    cv2.waitKey()


