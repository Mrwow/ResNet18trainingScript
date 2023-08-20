import cv2
import numpy as np
import os

def crop2square(img_in, img_out):
    image = cv2.imread(img_in)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    (cnts, _) = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image,[box], -1, (0,255,0),3)    #
    # window_name = 'image'
    # cv2.imshow(window_name,  image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs) if min(Xs) >= 0 else 0
    x2 = max(Xs)
    y1 = min(Ys) if min(Ys) >= 0 else 0
    y2 = max(Ys)
    h = y2 - y1
    w = x2 - x1
    s = max(h,w)
    cropImg = image[y1:y1+s, x1:x1+s, :]
    print(cropImg.shape)
    print(image.shape)
    cv2.imwrite(img_out, cropImg)

def crop2squareBatch(fd):
    dir = os.path.dirname(fd)
    out_fd = os.path.join(dir,"crop")
    if not os.path.exists(out_fd):
        os.mkdir(out_fd)

    for img in os.listdir(fd):
        if not img.startswith("."):
            img_in = os.path.join(fd,img)
            # img_out = img.replace(".png",'_crop.png')
            img_out = os.path.join(out_fd,img)
            crop2square(img_in,img_out)


def pad2square(img_in, img_out, pad_num):
    img = cv2.imread(img_in)
    h, w, c = img.shape
    print((w,h))
    if h >= w:
        s = h-w
        img_pad = cv2.copyMakeBorder(img,0,0,0,s,cv2.BORDER_CONSTANT,value=(pad_num,pad_num,pad_num))
    else:
        s = w - h
        img_pad = cv2.copyMakeBorder(img,0,s,0,0,cv2.BORDER_CONSTANT,value=(pad_num,pad_num,pad_num))
    cv2.imwrite(img_out, img_pad)

def pad2squareBatch(fd, pad_num):
    dir = os.path.dirname(fd)
    out_fd = os.path.join(dir,"pad")
    if not os.path.exists(out_fd):
        os.mkdir(out_fd)

    for img in os.listdir(fd):
        if not img.startswith("."):
            img_in = os.path.join(fd,img)
            # img_out = img.replace(".png",'_crop.png')
            img_out = os.path.join(out_fd,img)
            pad2square(img_in,img_out,pad_num)

def black2white(fd):
    for img in os.listdir(fd):
        if not img.startswith("."):
            img = os.path.join(fd,img)
            image = cv2.imread(img)
            img_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_ = 255 - img_
            cv2.imwrite(img,img_)

if __name__ == '__main__':
    # fd = "../source/test_img/seg"
    # crop2squareBatch(fd)
    # fd = "../source/test_img/crop"
    # pad2squareBatch(fd, pad_num=255)
    fd = "../split/train/0"
    pad2squareBatch(fd, pad_num=0)

    # img = "../test_crop/I_4-2-A10_1_seg.png"
    # crop2square(img)
    # pad2square(img)

    # fd = "../pad_w2b"
    # black2white(fd)
