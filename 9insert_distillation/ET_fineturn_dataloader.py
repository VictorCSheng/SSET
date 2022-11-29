import os
import random
from PIL import Image
import torch.utils.data as data
from tifffile import imread
import cv2
import numpy as np

#
def image_norm(img):
    if len(img.shape) == 2:
        imgmax = img.max()
        imgmin = img.min()
        imgnew = (img - imgmin) / (imgmax - imgmin)
    else:
        imgnew = np.zeros(img.shape)
        for i in range(3):
            imgmax = img[:,:,i].max()
            imgmin = img[:,:,i].min()
            imgnew[:,:,i] = (img[:,:,i] - imgmin) / (imgmax - imgmin)
    return imgnew

def read_frame_imgs(dir, frameorder, frameFlip_flag, fixsize, transform = None):
    imgfiles = os.listdir(dir)
    imgfiles = sorted(imgfiles)
    if len(imgfiles) < 3:
        print(dir)

    imgid = []
    imagestack = []
    flip_order = random.randint(0, 3)
    if frameorder == 0: # 正序
        for i in range(len(imgfiles)):
            imgidtemp = int(imgfiles[i][0:-4])
            imgid.append(imgidtemp)
            imgtemp = imread(os.path.join(dir, imgfiles[i]))
            if len(imgtemp.shape) == 2:
                imgtemp = np.expand_dims(imgtemp, axis=2)
                imgtemp = np.concatenate((imgtemp, imgtemp, imgtemp), axis=-1)
            # imgtemp = np.array(Image.open(os.path.join(dir, imgfiles[i])))
            # imgtemp = imread(os.path.join(dir, imgfiles[i]))
            imgtemp = image_norm(imgtemp)
            imgtemp = imgtemp.astype(np.float32)
            imgtemp = cv2.resize(imgtemp, fixsize, interpolation=cv2.INTER_AREA)
            if frameFlip_flag == 1:
                if flip_order == 0:     #
                    imgtemp = cv2.flip(imgtemp,-1)
                elif flip_order == 1:   #
                    imgtemp = cv2.flip(imgtemp, 0)
                elif flip_order == 2:   #
                    imgtemp = cv2.flip(imgtemp, 1)
                else:
                    imgtemp = imgtemp
            #
            # # imgtemp = cv2.cvtColor(imgtemp, cv2.COLOR_BGR2RGB)
            # imgtemp = Image.fromarray(imgtemp)  #
            # imgtemp = imgtemp.resize(fixsize, Image.ANTIALIAS)  # Image.ANTIALIAS
            # imgtemp = imgtemp.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else imgtemp
            # imgtemp = imgtemp.convert('RGB')

            if transform is not None:
                imgtemp = transform(imgtemp)

            # imgtemp = cv2.resize(imgtemp, fixsize, interpolation=cv2.INTER_AREA)
            imagestack.append(imgtemp)
        frame_t = (imgid[1] - imgid[0]) / (imgid[2] - imgid[0])
    else:
        for i in range(len(imgfiles) - 1, -1, -1):
            imgidtemp = int(imgfiles[i][0:-4])
            imgid.append(imgidtemp)
            imgtemp = imread(os.path.join(dir, imgfiles[i]))
            if len(imgtemp.shape) == 2:
                imgtemp = np.expand_dims(imgtemp, axis=2)
                imgtemp = np.concatenate((imgtemp, imgtemp, imgtemp), axis=-1)
            imgtemp = image_norm(imgtemp)
            imgtemp = imgtemp.astype(np.float32)
            imgtemp = cv2.resize(imgtemp, fixsize, interpolation=cv2.INTER_AREA)
            if frameFlip_flag == 1:
                if flip_order == 0:  #
                    imgtemp = cv2.flip(imgtemp, -1)
                elif flip_order == 1:  #
                    imgtemp = cv2.flip(imgtemp, 0)
                elif flip_order == 2:  #
                    imgtemp = cv2.flip(imgtemp, 1)
                else:
                    imgtemp = imgtemp

            if transform is not None:
                imgtemp = transform(imgtemp)

            imagestack.append(imgtemp)
        frame_t = (imgid[2] - imgid[1]) / (imgid[2] - imgid[0])
    return imagestack, frame_t

class ET(data.Dataset):
    def __init__(self, root, transform=None, fixsize=(352, 352), train=True):
        # frame in `root`.
        framesPath = os.listdir(root)
        framesPath = sorted(framesPath)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.transform = transform
        self.train = train

        self.fixsize = fixsize

        self.framesPath = framesPath

    def __len__(self):  #
        return len(self.framesPath)

    def __getitem__(self, index):
        if (self.train):
            ### Data Augmentation ###
            # Random reverse frame
            randomFrameorder = random.randint(0, 1)
            # Random flip frame
            randomFrameFlip_flag = 1
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            randomFrameorder = 0
            randomFrameFlip_flag = 0

        sample, frame_t = read_frame_imgs(os.path.join(self.root, self.framesPath[index]), randomFrameorder, randomFrameFlip_flag, self.fixsize, self.transform)

        return sample, frame_t  # 返

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
