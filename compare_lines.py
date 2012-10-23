import sys
import numpy as np
import cv2
from collections import defaultdict


def main(args):
    srcFolder = "/Users/huipeng/EO990RW8/"
    linesFile = open("/Users/huipeng/EO990RW8/lines_extract.txt", 'r')

    maxSizeX = 2880
    maxSizeY = 3520
    ranges = ((0, maxSizeX), (0, maxSizeY))
    bins = (maxSizeX // 32, maxSizeY // 32)
    total = bins[0] * bins[1] * 4

    ratio = 8
    templates = []
    templateImages = []
    templateMatchCounts = []

    maxSizeX /= ratio
    maxSizeY /= ratio

    headerLine = linesFile.readline()
    numImages = 0
    while len(headerLine) > 0:
        filename, numLines = headerLine.split()
        print filename
        lines = [[int(_) for _ in linesFile.readline().split()] for i in xrange(int(numLines))]
        lines = np.array(lines)
        isHorizontal = lines[:, 1] == lines[:, 3]
        horizontalLines = lines[isHorizontal]
        verticalLines = lines[np.invert(isHorizontal)]
        minX = np.min(horizontalLines[:, 0])
        minY = np.min(verticalLines[:, 1])
        lines[:, (0, 2)] -= minX
        lines[:, (1, 3)] -= minY

        lines /= ratio

        canvas = np.zeros((maxSizeY, maxSizeX))
        for line in lines:
            if np.all(line[[1, 3]] > maxSizeY /2):
#            if np.all(line[[0, 2]] < maxSizeX /2) and np.all(line[[1, 3]] > maxSizeY /2):
                continue
            cv2.line(canvas, tuple(line[0:2]), tuple(line[2:4]), (255, 255, 255), thickness=2)
        
        bestScore = 9999999999
        bestIndex = -1
        for i, template in enumerate(templates):
            score = np.sum(np.not_equal(template, canvas))
            if score < bestScore:
                bestScore = score
                bestIndex = i

        if bestScore > 7000:
            templateMatchCounts.append(1)
            templates.append(canvas)
            templateImages.append([(filename, canvas)])
        else:
            templateMatchCounts[bestIndex] += 1
            templateImages[bestIndex].append((filename, canvas))
#            print 'matached!'
#        cv2.imshow('ofo', canvas)
#        cv2.waitKey()

#
#        hLineStarts, dummy1, dummy2 = np.histogram2d(horizontalGridLines[:,0],
#                horizontalGridLines[:,1], bins, ranges)
#        hLineEnds, dummy1, dummy2 = np.histogram2d(horizontalGridLines[:,2],
#                horizontalGridLines[:,3], bins, ranges)
#        vLineStarts, dummy1, dummy2 = np.histogram2d(verticalGridLines[:,0],
#                verticalGridLines[:,1], bins, ranges)
#        vLineEnds, dummy1, dummy2 = np.histogram2d(verticalGridLines[:,2],
#                verticalGridLines[:,3], bins, ranges)
#
#        features = np.column_stack((
#                hLineStarts.flatten(),
#                hLineEnds.flatten(),
#                vLineStarts.flatten(),
#                vLineEnds.flatten())).flatten()
#        print len(features)

        headerLine = linesFile.readline()
        numImages += 1

    for i, imgGroup in enumerate(templateImages):
        for filename, img in imgGroup:
            cv2.imwrite('/Users/huipeng/templates/' + str(i) + '_' + filename + '.png', img)


    print 'num images', numImages
    print 'num templates', len(templateMatchCounts)
    print templateMatchCounts


if __name__ == '__main__':
    main(sys.argv)
