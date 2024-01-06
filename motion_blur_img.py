#根据原图生成模糊图片
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2

def average_blur(image):#平均模糊
    dst = cv2.blur(image, (5, 5))
    return dst
def median_blur(image):#中值模糊
    dst = cv2.medianBlur(image,5)
    return dst
def custom_blur(image):#自定义卷积核
    kernol = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 总和等于0或者总和等于1
    dst = cv2.filter2D(image, -1, kernol)
    return dst

#运动模糊CV2 to CV2
def motion_blur(image, degree=40, angle=90):
    #image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    print(motion_blur_kernel.shape)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
#交通标识模糊 CV2 TO CV2

import math

#单向的运动模糊
def single_blur(image,degree=40,angle=40):
    # 生成卷积核和锚点
    def genaratePsf(length, angle):
        EPS = np.finfo(float).eps
        alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
        cosalpha = math.cos(alpha)
        sinalpha = math.sin(alpha)
        if cosalpha < 0:
            xsign = -1
        elif angle == 90:
            xsign = 0
        else:
            xsign = 1
        psfwdt = 1
        # 模糊核大小
        sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
        sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
        psf1 = np.zeros((sy, sx))
        # psf1是左上角的权值较大，越往右下角权值越小的核。
        # 这时运动像是从右下角到左上角移动
        half = length / 2
        for i in range(0, sy):
            for j in range(0, sx):
                psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
                rad = math.sqrt(i * i + j * j)
                if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                    temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                    psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
                psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
                if psf1[i][j] < 0:
                    psf1[i][j] = 0
        # 运动方向是往左上运动，锚点在（0，0）
        anchor = (0, 0)
        # 运动方向是往右上角移动，锚点一个在右上角    #同时，左右翻转核函数，使得越靠近锚点，权值越大
        if angle < 90 and angle > 0:
            psf1 = np.fliplr(psf1)
            anchor = (psf1.shape[1] - 1, 0)
        elif angle > -90 and angle < 0:  # 同理：往右下角移动
            psf1 = np.flipud(psf1)
            psf1 = np.fliplr(psf1)
            anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
        elif angle < -90:  # 同理：往左下角移动
            psf1 = np.flipud(psf1)
            anchor = (0, psf1.shape[0] - 1)
        psf1 = psf1 / psf1.sum()
        return psf1, anchor

    kernel, anchor = genaratePsf(degree,angle)
    blur = cv2.filter2D(image, -1, kernel, anchor=anchor)
    return blur

#交通标识模糊，X部分组合的单向运动模糊进行融合
def traffic_blur(image,pieces = 9,angle_start = 0,angle_add = 90,degree=60,angle=45):

    def single_blur(image, degree=40, angle=40):
        # 生成卷积核和锚点
        def genaratePsf(length, angle):
            EPS = np.finfo(float).eps
            alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
            cosalpha = math.cos(alpha)
            sinalpha = math.sin(alpha)
            if cosalpha < 0:
                xsign = -1
            elif angle == 90:
                xsign = 0
            else:
                xsign = 1
            psfwdt = 1
            # 模糊核大小
            sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
            sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
            psf1 = np.zeros((sy, sx))
            # psf1是左上角的权值较大，越往右下角权值越小的核。
            # 这时运动像是从右下角到左上角移动
            half = length / 2
            for i in range(0, sy):
                for j in range(0, sx):
                    psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
                    rad = math.sqrt(i * i + j * j)
                    if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                        temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                        psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
                    psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
                    if psf1[i][j] < 0:
                        psf1[i][j] = 0
            # 运动方向是往左上运动，锚点在（0，0）
            anchor = (0, 0)
            # 运动方向是往右上角移动，锚点一个在右上角    #同时，左右翻转核函数，使得越靠近锚点，权值越大
            if angle < 90 and angle > 0:
                psf1 = np.fliplr(psf1)
                anchor = (psf1.shape[1] - 1, 0)
            elif angle > -90 and angle < 0:  # 同理：往右下角移动
                psf1 = np.flipud(psf1)
                psf1 = np.fliplr(psf1)
                anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
            elif angle < -90:  # 同理：往左下角移动
                psf1 = np.flipud(psf1)
                anchor = (0, psf1.shape[0] - 1)
            psf1 = psf1 / psf1.sum()
            return psf1, anchor

        kernel, anchor = genaratePsf(degree, angle)
        blur = cv2.filter2D(orig, -1, kernel, anchor=anchor)
        return blur

    # 生成8个方向的模糊核,从左到右，从上到下依次编号
    mydegree = degree

    blur3 = single_blur(orig, degree=mydegree, angle=45)
    blur1 = single_blur(orig, degree=mydegree, angle=135)
    blur9 = single_blur(orig, degree=mydegree, angle=-45)
    blur7 = single_blur(orig, degree=mydegree, angle=-135)

    blur6 = single_blur(orig, degree=mydegree, angle=1)
    blur2 = single_blur(orig, degree=mydegree, angle=91)
    blur8 = single_blur(orig, degree=mydegree, angle=-89)
    blur4 = single_blur(orig, degree=mydegree, angle=-179)

    blur51 = motion_blur(orig, degree=20, angle=90)
    blur52 = motion_blur(orig, degree=20, angle=-90)
    blur5 = average_blur(orig)

    blurlist = [blur1, blur2, blur3, blur4, blur5, blur6, blur7, blur8, blur9]


    canvas_shape = orig.shape
    canvas_color = 0, 0, 0
    canvas_1 = np.ones(canvas_shape, dtype=np.uint8)
    canvas_1[:] = canvas_color

    path = canvas_shape[0] // 3

    curnum = -1
    for m in range(3):
        for n in range(3):
            curnum += 1
            curblur = blurlist[curnum]
            for i in range(m * path, (m + 1) * path):
                for j in range(n * path, (n + 1) * path):
                    canvas_1[i, j] += curblur[i][j]
    blur = canvas_1
    return blur




if __name__ == "__main__":
    image_path="test/original/000000033109.jpg"
    blur_path = 'output/blur_images/'
    orig = cv2.imread(image_path)[..., ::-1]
    #orig = cv2.resize(orig, (224, 224))
    b, g, r = cv2.split(orig)
    orig= cv2.merge([r, g, b])

    blur_sb = single_blur(orig,degree=20,angle=-45) #  单向运动模糊生成

    mydegree = 30

    blur3 = single_blur(orig,degree=mydegree,angle=45)
    blur1 = single_blur(orig,degree=mydegree,angle=135)
    blur9 = single_blur(orig,degree=mydegree,angle=-45)
    blur7 = single_blur(orig,degree=mydegree,angle=-135)

    blur6 = single_blur(orig,degree=mydegree,angle=1)
    blur2 = single_blur(orig,degree=mydegree,angle=91)
    blur8 = single_blur(orig,degree=mydegree,angle=-89)
    blur4 = single_blur(orig,degree=mydegree,angle=-179)

    blur51 = motion_blur(orig, degree=10, angle=90)
    blur52 = motion_blur(orig, degree=10, angle=-90)
    blur5 = average_blur(orig)


    blurlist=[blur1,blur2,blur3,blur4,blur5,blur6,blur7,blur8,blur9]
    print(blur3.shape)

    #原图大小，值为0的画布
    canvas_shape= orig.shape
    canvas_color = 0,0,0
    canvas = np.ones(canvas_shape, dtype=np.uint8)
    canvas[:] = canvas_color

    #从右上角开始，逆时针把四块模糊放到画布上
    #CV2图片左上角是0,0 右下角是最大的
    canvas_shape_half = canvas_shape[0]//2
    half,max = canvas_shape_half+1,canvas_shape[0]
    x,y = [1,2],[1,2]
    #左上
    for i in range(0,half):
        for j in range(0,half):
            canvas[i][j] += blur1[i][j]
    #右上
    for i in range(0,half):
        for j in range(half,max):
            canvas[i][j] += blur1[i][j]
    #左下
    for i in range(half,max):
        for j in range(0,half):
            canvas[i][j] += blur3[i][j]
    #右下
    for i in range(half,max):
        for j in range(half,max):
            canvas[i][j] += blur2[i][j]
    t_blur=traffic_blur(orig)
    canvas_shape = orig.shape
    canvas_color = 0, 0, 0
    canvas_1 = np.ones(canvas_shape, dtype=np.uint8)
    canvas_1[:] = canvas_color
    path = canvas_shape[0]//3
    curnum=-1
    for m in range(3):
        for n in range(3):
            curnum +=1
            curblur = blurlist[curnum]
            for i in range(m*path,(m+1)*path):
                for j in range(n*path,(n+1)*path):
                    canvas_1[i,j] += curblur[i][j]

    # cv2.imshow('X2',blur_sb)  #  显示单向运动模糊图像
    cv2.imwrite(blur_path+'blur_sb.jpg',blur_sb)   #  保存单向运动模糊图像

    cv2.waitKey()



