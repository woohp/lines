import sys
import os
import glob
import subprocess
import shutil
import cPickle as pickle
from itertools import izip
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import RandomizedPCA
from score import score
from classifier_extensions import pick_best_svc_classifier


# the cache of features
# the key is (filename, (canvas_size, grid_size))
# the value is the features
features_cache = None


def cluster_documents(
        filepaths,
        canvas_size=(800,650),
        grid_size=64,
        normalize_image=True,
        score_weights={ 'accuracy': 0.6, 'purity': 0.3, 'nmi': 0.1 },
        dbscan_eps=270,
        dbscan_min_samples=2,
        pca_n_components=100,
        output_path=None):

    global features_cache
    try:
        with open('features.pkl', 'r') as f:
            features_cache = pickle.load(f)
    except:
        features_cache = {}

    key = (canvas_size, grid_size)
    if key not in features_cache:
        features_cache[key] = {}

    # calculate the features
    filenames = []
    X = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        filenames.append(filename)

        if filename not in features_cache[key]:
            height, width = canvas_size
            lines_filepath = os.path.join(os.getcwd(), 'lines')
            pipe = subprocess.Popen(
                    [lines_filepath, filepath, str(height), str(width), str(grid_size), str(int(normalize_image))],
                    stdout=subprocess.PIPE)
            out, err = pipe.communicate()
            features_cache[key][filename] = np.fromstring(out.strip(), dtype=int, sep=' ')

        X.append(features_cache[key][filename])
    pickle.dump(features_cache, open('features.pkl', 'w'))
    print 'finished calculating features'

    # do the clustering using DBSCAN
    dbscan = DBSCAN(eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            random_state=np.random.RandomState(0))
    dbscan.fit(np.atleast_2d(X))

    # calcualte some statistics from the clustering
    num_clusters = len(set(dbscan.labels_)) - 1
    num_unclustered = sum(1 for x in dbscan.labels_ if x == -1)
    if num_clusters < 3:
        print 'did not have enough clusters. returning'
        return
    print 'num clusters:', num_clusters
    print 'num unclustered:', num_unclustered

    # calculate score
    cluster_counter, unestimated, purity, nmi, accuracy = score(izip((str(int(l)) for l in dbscan.labels_), filenames))
    combined_score = score_weights['accuracy'] * accuracy + score_weights['purity'] * purity + score_weights['nmi'] * nmi
    print 'scores:\n\tpurity: %f\n\tnmi: %f\n\taccuracy: %f\n\tcombined: %f' % (purity, nmi, accuracy, combined_score)


    # separate data into labeled and unlabeled lists
    labeled_X = []
    labeled_Y = []
    labeled_filenames = []
    unlabeled_X = []
    unlabeled_filenames = []
    for x, y, filename in izip(X, dbscan.labels_, filenames):
        if y == -1:
            unlabeled_X.append(x)
            unlabeled_filenames.append(filename)
        else:
            labeled_X.append(x)
            labeled_Y.append(y)
            labeled_filenames.append(filename)

    # fit PCA
    pca = RandomizedPCA(n_components=pca_n_components)
    pca.fit(X)
    labeled_X = pca.transform(labeled_X)
    unlabeled_X = pca.transform(unlabeled_X)

    # train classifier
    kernels = ['linear', 'rbf']
    Cs = [2 ** i for i in xrange(7)]
    gammas = [2.0 ** i for i in xrange(-6, 2)]
    clf = pick_best_svc_classifier(kernels, Cs, gammas, labeled_X, labeled_Y, k=2)
    clf.fit(labeled_X, labeled_Y)

    # predict the unlabeled data
    predicted_Y = clf.predict(unlabeled_X).tolist()
    all_Y = labeled_Y + predicted_Y
    all_filenames = labeled_filenames + unlabeled_filenames
    cluster_counter, unestimated, purity, nmi, accuracy = score(izip((str(int(y)) for y in all_Y), all_filenames), izip(predicted_Y, unlabeled_filenames))

    combined_score_after_classification = score_weights['accuracy'] * accuracy + score_weights['purity'] * purity + score_weights['nmi'] * nmi
    print 'scores after classification:\n\tpurity: %f\n\tnmi: %f\n\taccuracy: %f\n\tcombined: %f' % (purity, nmi, accuracy, combined_score_after_classification)

    if output_path:
        template_path = os.path.join(output_path, 'templates')

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        os.mkdir(template_path)
        for label, filepath in izip(dbscan.labels_, filepaths):
            label = str(int(label))

            group_path = os.path.join(output_path, label)
            is_first_instance = not os.path.isdir(group_path)
            if is_first_instance:
                os.mkdir(group_path)

            filename = os.path.basename(filepath)
            subprocess.call(['cp', filepath, os.path.join(group_path, filename)])
            if is_first_instance and label != '-1':
                subprocess.call(['cp', filepath, os.path.join(template_path, label + '_' + filename)])
    


def main(argv):
    filepaths = glob.glob(os.path.join(os.getcwd(), 'unlabeled', '*'))
    cluster_documents(filepaths, output_path=os.path.join(os.getcwd(), 'groups'))


if __name__ == '__main__':
    main(sys.argv)

