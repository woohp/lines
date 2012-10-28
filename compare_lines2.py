import sys
import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def main(args):
    linesFile = sys.stdin
    if len(args) > 1:
        linesFile = open(args[1], 'r')

    allFeatures = []
    allFilenames = []
    filename = linesFile.readline()
    while len(filename) > 0:
        dataStr = linesFile.readline()
        features = np.fromstring(dataStr, dtype=int, sep=' ')
        features = features[:len(features)//2]

        allFeatures.append(features)
        allFilenames.append(filename.strip())

        filename = linesFile.readline()
    print 'finished reading all filenames. clustering...'


    dbscan = DBSCAN(eps=1100, min_samples=2, random_state=np.random.RandomState(0))
    dbscan.fit(np.atleast_2d(allFeatures))
    print 'num clusters', len(set(dbscan.labels_))


    shutil.rmtree('/Users/huipeng/groups/')
    os.mkdir('/Users/huipeng/groups/')
    os.mkdir('/Users/huipeng/groups/templates/')
    for label, filename in zip(dbscan.labels_, allFilenames):
        label = str(int(label))

        groupFolder = os.path.join('/Users/huipeng/groups/', label)
        isFirstInstance = not os.path.isdir(groupFolder)
        if isFirstInstance:
            os.mkdir(groupFolder)

        originalImage = cv2.imread(os.path.join("/Users/huipeng/EO990RW8/", filename), 0)
        height, width = originalImage.shape
        resizedImage = cv2.resize(originalImage, (width/4, height/4))

        newFilename = os.path.join(groupFolder, filename + '.png')
        cv2.imwrite(newFilename, resizedImage)
        if isFirstInstance and label != '-1':
            newFilename = os.path.join('/Users/huipeng/groups/templates/', label + '_' + filename + '.png')
            cv2.imwrite(newFilename, resizedImage)

    print 'finished'



if __name__ == '__main__':
    main(sys.argv)
