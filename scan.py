import os

import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

'''
定义图片展示函数 和 图片大小设置函数
'''
def show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#为了方便统一图片大小
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized
'''
读取输入图片，做预处理
'''
#读取图片进来
#img = cv.imread(r'C:\images\receipt.jpg')
img = cv.imread(r'C:\images\page.jpg')

org = img.copy()
#show('img',img)
#统一图片大小
img = resize(img,height=500)
#记录比例
ratio = org.shape[0]/500.0
print(f'radio={ratio}')
#转化为灰度图
img_gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#高斯滤波降噪
img_gray = cv.GaussianBlur(img_gray,(5,5),0)
#show('gray',img_gray)


'''
边缘检测
由于图像边缘非常容易受到噪声的干扰，因此为了避免检测到错误的边缘信息，通常需要对图像进行滤波以去除噪声。滤波的目的是平滑一些纹理较弱的非边缘区域，以便得到更准确的边缘。在实际处理过程中，通常采用高斯滤波去除图像中的噪声。
在滤波过程中，我们通过滤波器对像素点周围的像素计算加权平均值，获取最终滤波结果。滤波器的大小也是可变的，高斯核的大小对于边缘检测的效果具有很重要的作用。滤波器
的核越大，边缘信息对于噪声的敏感度就越低。不过，核越大，边缘检测的定位错误也会随之增加。通常来说，一个 5×5 的核能够满足大多数的情况。

'''
edged = cv.Canny(img_gray,100,200)
show("edged",edged)

#轮廓检测
cnts = cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[1]
img_cnts = cv.drawContours(img.copy(),cnts,-1,(0,0,255),2)
show('',img_cnts)

#轮廓排序 取面积最大的几个
need_cnts = sorted(cnts,key=cv.contourArea,reverse=True)[0:10]
# print(len(need_cnts))
# cv.drawContours(img, need_cnts, 3, (0, 255, 0), 2)
# show('Outline',img)

#便利轮廓
for (i,c) in enumerate(need_cnts):
    #轮廓相似
    #计算轮廓长度
    length = cv.arcLength(c,True)#Ture代表轮廓是闭合的
    #轮廓拟合：第一个参数是轮廓，第二个参数是精度，精度越高，拟合效果越好，Ture表示轮廓是闭合的
    approx = cv.approxPolyDP(c,0.01*length,True)
    if len(approx)==4:#如果轮廓有四个顶点
        screenCnt = approx
        break
#轮廓点的坐标为[横坐标，纵坐标]
cv.drawContours(img, [screenCnt], 0, (0, 255, 0), 5)
show('screenCnt', img)
pts = screenCnt.reshape(4,2)
print(pts)

#把坐标分为左上，右上，右下，左下
average_x = float(sum(pts[:,0])/4)
print(average_x)
average_y = float(sum(pts[:,1])/4)
print(average_y)
rect = np.zeros((4, 2), dtype="float32")
print(type(rect))
for (i,pos) in enumerate(pts):
    if pos[0]<=average_x and pos[1]<=average_y:
        tl = np.float32(pos*ratio)
    elif pos[0]>=average_x and pos[1]<=average_y:
        tr = np.float32(pos*ratio)
    elif pos[0]>=average_x and pos[1]>=average_y:
        br = np.float32(pos*ratio)
    else:
        bl = np.float32(pos*ratio)

rect = tl,tr,br,bl

#计算最大边长
width1 = np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)
width2 = np.sqrt((br[0]-bl[0])**2+(br[1]-bl[1])**2)
width_max = max(int(width1),int(width2))
hight1 = np.sqrt((tl[0]-bl[0])**2+(tl[1]-bl[1])**2)
hight2 = np.sqrt((tr[0]-br[0])**2+(tr[1]-br[1])**2)
hight_max = max(int(hight1),int(hight2))

#变换后的坐标
dst = np.array([
    [0,0],
    [width_max-1,0],
    [width_max-1,hight_max-1],
    [0,hight_max-1]],dtype="float32")

print(dst)
print(type(dst))
rect = np.array(rect,dtype='float32')
print(rect)
print(type(rect))
#计算变化矩阵
M = cv.getPerspectiveTransform(rect,dst)
print(M)
#变换后的图像
warped = cv.warpPerspective(org,M,(width_max,hight_max))
warped = cv.resize(warped,dsize=None,fx=0.5,fy=0.5)
show("warped",warped)

'''
文本检测
'''
#把图像转化为灰度图
warped_gray = cv.cvtColor(warped,cv.COLOR_BGR2GRAY)
show('waeped_gray',warped_gray)

#滤波
warped_gray = cv.medianBlur(warped_gray,3)#3是卷积核大小
show('median',warped_gray)

#把最终的图片创建一个文件名，使Image.open这个函数能打开它
filename = "{}.png".format('finally_picture')
cv.imwrite(filename,warped_gray)
#text = pytesseract.image_to_string(Image.open(filename), lang='chi_sim')
text = pytesseract.image_to_string(Image.open(filename))
print(text)
#删除文件
os.remove(filename)



