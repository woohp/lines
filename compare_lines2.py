import sys
import os
import numpy as np
import cv2
from collections import defaultdict
from subprocess import call
from sklearn.cluster import DBSCAN


def main(args):
    srcFolder = "/Users/huipeng/EO990RW8/"
    linesFile = open("/Users/huipeng/EO990RW8/lines_extract.txt", 'r')

    maxSizeX = 2880
    maxSizeY = 3520

    headerLine = linesFile.readline()
    templates = []
    templateMatches = []
    allfeatures = []
    allfilenames = []
    while len(headerLine) > 0:
        filename = headerLine.split()[0]
        print filename
        dataStr = linesFile.readline()
        features = np.fromstring(dataStr, dtype=int, sep=' ')
        features = features[:len(features)//2]

#        bestScore = 0
#        bestIndex = -1
#        for i, template in enumerate(templates):
#            score = np.dot(features, template)
#            if score > bestScore:
#                bestScore = score
#                bestIndex = i
#            bestMatch = max(bestScore, score)
#
#        print 'bestScore', bestScore, bestIndex 
#        if bestScore > 100:
#            templateMatches[bestIndex].append(filename)
#        else:
#            templates.append(features)
#            templateMatches.append([filename])

        allfilenames.append(filename)
        allfeatures.append(features)

        headerLine = linesFile.readline()

    dbscan = DBSCAN(eps=1200, min_samples=2)
    dbscan.fit(np.array(allfeatures))

    call(['rm', '-rf', '/Users/huipeng/groups/'])
    call(['mkdir', '/Users/huipeng/groups/'])
    for label, filename in zip(dbscan.labels_, allfilenames):
        call(['mkdir', '/Users/huipeng/groups/' + str(label)])
        originalImage = cv2.imread('/Users/huipeng/EO990RW8/' + filename, 0)
        height, width = originalImage.shape
        resizedImage = cv2.resize(originalImage, (width/4, height/4))
        cv2.imwrite('/Users/huipeng/groups/' + str(label) + '/' + filename + '.png', resizedImage)


    print 'dbscan num clusters', zip(dbscan.labels_, allfilenames)

#    for group in templateMatches:
#        print len(group), group
#    print len(templateMatches)


if __name__ == '__main__':
    main(sys.argv)
