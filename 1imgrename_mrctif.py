import os

def dir_img_rename(imgdir):
    imglist = os.listdir(imgdir)
    imglist = sorted(imglist)

    for i in range(len(imglist)):
        imgoldname = imgdir + imglist[i]
        imgnewname = imgdir + str(i).zfill(4) + '.tif'
        try:
            os.rename(imgoldname, imgnewname)
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        else:
            print('rename file success\r\n')

if __name__ == '__main__':
    dir_dir = 'E:/Zhu/ETreg/newdata/HUA3/TOM/stack/ori/'
    dirnamelist = os.listdir(dir_dir)
    for i in range(len(dirnamelist)):
        imgdir = dir_dir + dirnamelist[i] + '/'
        dir_img_rename(imgdir)