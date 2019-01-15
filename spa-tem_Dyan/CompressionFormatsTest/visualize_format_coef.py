import os, sys
from PIL import Image
import matplotlib
import jpeg_toolbox_python.python.jpegtools

folderList = ['set10V011']
rootDir = '/home/armandcomas/datasets/Caltech/images'
for folder in folderList:
    frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if
              each.endswith(('.jpg', '.jpeg', '.bmp', 'png'))]
    frames.sort()
    for i in range(25, 100, 1):
        imgname = os.path.join(rootDir, folder, frames[i])
        img = Image.open(imgname)
        raw = list(img.getdata())
