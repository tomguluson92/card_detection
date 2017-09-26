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


def image_process(img):
    # 1、读取图像，并把图像转换为灰度图像并显示
    img = cv2.imread(img)
    grey = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(grey, 80, 150)


    # 2、注意条形码区域是怎样通过梯度操作检测出来的。下一步将通过去噪仅关注条形码区域。
    # 我们要做的第一件事是使用4 * 4的内核对梯度图进行平均模糊，
    blurred = cv2.blur(canny, (4, 4))
    retval, grey = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)


    # 3、画出轮廓 cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.findContours 返回结构：图像，轮廓，层析结构
    image, contours, hierarchy = cv2.findContours(grey.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # # # # 0,255,0 绿色
    cv2.drawContours(img, [box], -1, (0, 0, 255), 3)
    # # # 第三个参数为-1表示打印所有轮廓
    cv2.imshow('Image', img)
    # cv2.imwrite("./processed.jpg", newimg)
    cv2.waitKey()

if __name__ == "__main__":
    image_process('./test.jpg')
