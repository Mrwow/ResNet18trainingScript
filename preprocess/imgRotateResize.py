import os
import argparse
import random

import torchvision.transforms
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt


def img_aug(img,aug):
    out = img.replace(".png", "aug.png")
    img = Image.open(img)
    w, h = img.size
    img = torchvision.transforms.ToTensor()(img)
    plt.subplot(1,2,1)
    plt.imshow(img.T.transpose(1,0))

    img_aug = aug(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_aug.T.transpose(1,0))
    plt.show()

def random_rotate_scaling(img, img_out):
    img = Image.open(img)
    print(img.size)
    n_max = max(img.size)
    n_min = min(img.size)

    aug_pad = transforms.Pad([0, 0, (n_max - n_min), 0])
    aug_rotate = transforms.RandomRotation((-180, 180))
    aug = transforms.Compose([aug_pad, aug_rotate])
    img_rotate = aug(img)

    w, h = img_rotate.size
    print(img_rotate.size)
    scale_num = random.randint(50, 150)/100
    aug_resize = transforms.Resize((round(w*scale_num), round(h*scale_num)))
    img_aug = aug_resize(img_rotate)
    print(img_aug.size)
    img_aug.save(img_out)

def aug_flip(img, img_out_h, img_out_v):
    img = Image.open(img)
    aug_flip_h = transforms.RandomHorizontalFlip(p=1)
    img_flig_h =aug_flip_h(img)
    img_flig_h.save(img_out_h)

    aug_flip_v = transforms.RandomVerticalFlip(p=1)
    img_flig_v = aug_flip_v(img)
    img_flig_v.save(img_out_v)


def batch_random_rotate_scaling(fd, num=3):
    # first do flip

    for img in os.listdir(fd):
        if not img.startswith("."):
            img_url = os.path.join(fd, img)
            img_out_h = img.replace(".", "_h.")
            img_out_h = os.path.join(fd, img_out_h)
            img_out_v = img.replace(".", "_v.")
            img_out_v = os.path.join(fd, img_out_v)
            aug_flip(img_url,img_out_h,img_out_v)

    # second do rotation and resize
    for img in os.listdir(fd):
        if not img.startswith("."):
            # print(img)
            img_url = os.path.join(fd, img)
            for i in range(int(num)):
                n = "_" + str(i) + "."
                img_out = img.replace(".", n)
                # print(img_out)
                img_out_url = os.path.join(fd, img_out)
                # print(img_out_url)
                random_rotate_scaling(img_url, img_out_url)




parser = argparse.ArgumentParser(description="Define the data folder, augment number")
parser.add_argument('--fd', '-f', help='image folder')
parser.add_argument('--number', '-n', help='augment number')
args = parser.parse_args()

if __name__ == '__main__':

    batch_random_rotate_scaling(args.fd, args.number)



