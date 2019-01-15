import os
import hickle as hkl
import numpy as np

rootDir = '/home/armandcomas/datasets/Kitti/'
training_file = 'X_train.hkl'
sources_file = 'sources_train.hkl'

def getIndex(sources_list):
    index = {}
    folderName = set(sources_list)
    for folder in folderName:
        first = 0
        last = 0
        for count, name in enumerate(sources_list):
            if folder == name:
                if first == 0:
                    if count == 0:
                        first = count - 1
                    else:
                        first = count

                last = count
                index[folder] = {"first": first, "last": last + 1}
            else:
                continue

    if index['city-2011_09_26_drive_0001_sync']['first'] == -1:
        index['city-2011_09_26_drive_0001_sync']['first'] = 0

    return index

# imageArray = hkl.load(os.path.join(rootDir, training_file))
# sources_list = hkl.load(os.path.join(rootDir, sources_file))
# indexArray = getIndex(sources_list)
# listOfFolders = [key for key in indexArray]
#
# print(listOfFolders)

# trainingData = videoDataset(listOfFolders=listOfFolders,
#                                                         imageArray=imageArray,
#                                                         index=indexArray,
#                                                         N_FRAME=N_FRAME,
#                                                         imageSize=(128,160),
#                                                         channel = 0)

imageArray = hkl.load(os.path.join(rootDir, training_file))
sources_list = hkl.load(os.path.join(rootDir, sources_file))

indexArray = getIndex(sources_list)
listOfFolders = [key for key in indexArray]

for folder in listOfFolders:
    value = indexArray[folder]
    nFrames = value['last'] - value['first']
    print(nFrames, value, folder)
    numBatches = min(int(nFrames/11),1)
    for batchnum in range(numBatches):
        for framenum in range(11):
                img = imageArray[framenum + 11*batchnum + value['first']]
                if not os.path.exists(os.path.join("/home/armandcomas/datasets/Kitti/frames", folder)):
                        os.makedirs(os.path.join("/home/armandcomas/datasets/Kitti/frames", folder))
                np.save(os.path.join("/home/armandcomas/datasets/Kitti/frames", folder,str(framenum)+'.npy'), img)
