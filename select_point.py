import cv2

ref_pointx = []
ref_pointy = []
def click(event, x,y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pointx.append(x)
        ref_pointy.append(y)
        with open("Homopoint2d_Right.txt", "w") as f:
            f.write("{}".format(ref_pointx))
            f.write("{}".format(ref_pointy))


window = cv2.namedWindow("select_point")
cv2.setMouseCallback("select_point", click)
image = cv2.imread("/home/wang_shuai/Assignment-4-Material/Right.jpg")
cv2.imshow("select_point", image)
cv2.waitKey()