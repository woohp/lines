import sys
import glob
import cv2

def main(args):
    feature_detector = cv2.FeatureDetector_create('SIFT')
    descriptor_extractor = cv2.DescriptorExtractor_create('SIFT')
    matcher = cv2.DescriptorMatcher_create('FlannBased')

    descriptors = []
    distance_matrix = []

    filenames = glob.glob('/Users/huipeng/Desktop/unlabeled/*')
    i = 0
    for filename in filenames:
        img = cv2.imread(filename, 0)
        height, width = img.shape
        img = cv2.resize(img, (int(width//2), int(height//2)))

        keypoints = feature_detector.detect(img)
        keypoints, descriptor = descriptor_extractor.compute(img, keypoints)
        descriptors.append(descriptor)

        i += 1


    for i in xrange(len(descriptors)):
        distances = []
        for j in xrange(i+1, len(descriptors)):
            matches = matcher.match(descriptors[i], descriptors[j])
            distance = sum(match.distance for match in matches)
            distances.append(distance)

        distance_matrix.append(distances)

    print distance_matrix

#        cv2.namedWindow('foo')
#        cv2.imshow('foo', img)
#        cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
