import os
import shutil
from csutil import util


if __name__ == "__main__":
    gap_num = [37, 40, 80, 51, 39, 89, 45, 52, 56]

    root_dir = 'E:/Zhu/ETreg/newdata/HUA3/'

    data_dir = root_dir + '/TOM/stack/result/'
    resultdir = root_dir + '/TOM/stack/Finaltemp/'
    util.mkdir(resultdir)

    imgid = 0
    img_dirs = os.listdir(data_dir)
    img_dirs.sort(key=lambda x: int(x))
    for i in range(len(img_dirs)):
        img_dir = data_dir + img_dirs[i] + "/"
        img_files = os.listdir(img_dir)
        img_files.sort(key=lambda x: int(x[0:-4]))
        if i != 0:
            imgid = imgid + gap_num[i-1]
        for j in range(len(img_files)):
            shutil.copyfile(img_dir + img_files[j],
                            os.path.join(resultdir, (str(imgid).zfill(4) + ".tif")))
            imgid = imgid + 1
