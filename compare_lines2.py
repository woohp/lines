import sys
import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from score import score 
from classifier_extensions import pick_best_svc_classifier


def main(args):
    print 'here2'
    store_cluster = False
    linesFile = sys.stdin
    if len(args) > 1:
        linesFile = open(args[1], 'r')
    if len(args) > 2:
        if args[2] == '-store':
            store_cluster = True

    allFeatures = []
    allFilenames = []
    filename = linesFile.readline()
    while len(filename) > 0:
        dataStr = linesFile.readline()
        features = np.fromstring(dataStr, dtype=int, sep=' ')

        allFeatures.append(features)
        allFilenames.append(filename.strip())

        filename = linesFile.readline()
    print 'finished reading all filenames. clustering...'


    best_cluster = None
    for eps in range(150, 400, 10):
        print eps
        dbscan = DBSCAN(eps=eps, min_samples=2, random_state=np.random.RandomState(0))
        dbscan.fit(np.atleast_2d(allFeatures))
        print 'num clusters', len(set(dbscan.labels_))
        if len(set(dbscan.labels_)) < 3:
            continue

        cluster_counter, unestimated, purity, nmi, accuracy = score(zip([str(int(l)) for l in dbscan.labels_], allFilenames))
        combined_score = 0.6 * accuracy + 0.3 * purity + 0.1 * nmi
        print combined_score, purity, nmi, accuracy
        print unestimated
        print
        if best_cluster is None or best_cluster[0] < combined_score:
            best_cluster = (combined_score, dbscan.labels_, eps)
    print best_cluster[0], best_cluster[2]


    labeled_X = []
    labeled_Y = []
    unlabeled_X = []
    labeled_fname = []
    unlabeled_fname = []
    for fname, x, y in zip(allFilenames, allFeatures, best_cluster[1]):
        if y == -1:
            unlabeled_X.append(x)
            unlabeled_fname.append(fname)
        else:
            labeled_X.append(x)
            labeled_Y.append(y)
            labeled_fname.append(fname)


    pca = RandomizedPCA(n_components=100)
    all_X = np.concatenate((labeled_X, unlabeled_X))
    pca.fit(all_X)
    labeled_X = pca.transform(labeled_X)
    unlabeled_X = pca.transform(unlabeled_X)
    
    kernels = ['linear', 'rbf']
    Cs = [2 ** i for i in xrange(7)]
    gammas = [2.0 ** i for i in xrange(-6, 2)]
    clf = pick_best_svc_classifier(kernels, Cs, gammas, labeled_X, labeled_Y, k=2)
    clf.fit(labeled_X, labeled_Y)

    predicted_Y = clf.predict(unlabeled_X).tolist()
    all_Y = labeled_Y + predicted_Y
    all_fnames = labeled_fname + unlabeled_fname
    cluster_counter, unestimated, purity, nmi, accuracy = score(zip([str(int(y)) for y in all_Y], all_fnames), zip(predicted_Y, unlabeled_fname))
    print predicted_Y
    print purity
    print nmi
    print accuracy

    if not store_cluster:
        return

    homeDir = os.path.expanduser('~')
    groupsDir = os.path.join(homeDir, 'groups')
    templatesDir = os.path.join(groupsDir, 'templates')

    shutil.rmtree(groupsDir)
    os.mkdir(groupsDir)
    os.mkdir(templatesDir)
    for label, filename in zip(best_cluster[1], allFilenames):
        label = str(int(label))

        groupFolder = os.path.join(groupsDir, label)
        isFirstInstance = not os.path.isdir(groupFolder)
        if isFirstInstance:
            os.mkdir(groupFolder)

        originalImage = cv2.imread(os.path.join("/Users/huipeng/Desktop/unlabeled/", filename), 0)
        height, width = originalImage.shape
        #resizedImage = cv2.resize(originalImage, (width/4, height/4))
        resizedImage = originalImage

        newFilename = os.path.join(groupFolder, os.path.basename(filename))
        cv2.imwrite(newFilename, resizedImage)
        if isFirstInstance and label != '-1':
            newFilename = os.path.join(templatesDir, label + '_' + os.path.basename(filename))
            cv2.imwrite(newFilename, resizedImage)

    print 'finished'



if __name__ == '__main__':
    main(sys.argv)
