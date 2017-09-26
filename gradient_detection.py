# coding: UTF-8

'''
    @function:auto detection of bank card magnetic stripe
    @author: samuel gao
    @institute: airdoc
    @date: 2017/9/26
'''

import numpy as np
import datetime
import time
import cv2


def sobel_process(img):
    # 1、读取图像，并把图像转换为灰度图像并显示
    img = cv2.imread(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradY, gradX)
    gradient = cv2.convertScaleAbs(gradient)

    # 2、注意条形码区域是怎样通过梯度操作检测出来的。下一步将通过去噪仅关注条形码区域。
    # 我们要做的第一件事是使用16 * 16的内核对梯度图进行平均模糊，这里越大越好
    blurred = cv2.blur(gradient, (16, 16))
    retval, grey = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)

    # 3、形态学处理
    # 形态学腐蚀
    closed = cv2.erode(closed, None, iterations=5)
    # 形态学膨胀
    grey = cv2.dilate(closed, None, iterations=5)

    # 4、画出轮廓
    image, contours, hierarchy = cv2.findContours(grey.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    cv2.drawContours(img, [box], -1, (0, 0, 255), 3)
    # # # 第三个参数为-1表示打印所有轮廓
    cv2.imshow('Image', img)

    cv2.waitKey()

# 颜色区域提取
def color_area(work_hsv):
    img = cv2.imread(work_hsv)
    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换了灰度化
    # 提取黑色区域
    low = np.array([0, 0, 0])
    high = np.array([100, 100, 100])
    mask = cv2.inRange(img, low, high)
    black = cv2.bitwise_and(img, img, mask=mask)
    return black

if __name__ == "__main__":
    sobel_process('./test.jpg')