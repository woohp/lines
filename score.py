import os
import glob
import math
import scipy
from scipy.misc import comb
from collections import Counter
from itertools import groupby

TRUE_LABEL_DIR = '/Users/huipeng/lines/labeled/'
ESTIMATED_LABEL_DIR = '/Users/huipeng/lines/groups/'

true_labels = glob.glob(os.path.join(TRUE_LABEL_DIR, '*'))
lookup = {}
for true_label in true_labels:
    for example in glob.glob(os.path.join(true_label, '*')):
        lookup[os.path.basename(example)] = os.path.basename(true_label)

def score(clusters, classifier_ex=None):
    label_lookup = {}

    unestimated = 0
    purity_count = 0.0
    total = 0.0
    label_counter = Counter()
    cluster_counter = []
    clusters = groupby(sorted(clusters, key=lambda x: x[0]), key=lambda x: x[0])
    for label, cluster in clusters:
        if label == '-1':
            unestimated = len(list(cluster))
            continue

        counter = Counter()
        for estimated_label, ele in cluster:
            counter[lookup[os.path.basename(ele)]] += 1.0
        cluster_counter.append((label, counter))

        cluster_count = sum(counter.values())
        most_common = counter.most_common(1)[0]
        label_lookup[label] = most_common[0]
        purity_count += most_common[1]
        total += cluster_count
        label_counter += counter

    if classifier_ex:
        classifier_total = 0
        classifier_correct = 0.0
        for predicted_label, ex in classifier_ex:
            true_label = lookup[os.path.basename(ex)]
            classifier_correct += 1 if label_lookup[str(int(predicted_label))] == true_label else 0
            classifier_total += 1
        print 'Classifier Accuracy', classifier_correct / classifier_total

    I = 0
    H_O = 0
    total_positives = 0
    true_positives = 0
    total_negatives = 0
    true_negatives = 0
    for i, (cluster_label, cluster) in enumerate(cluster_counter):
        # Calucating "Mutual information"
        temp_I = 0
        cluster_count = sum(cluster.values())
        for key in cluster.keys(): # key is the label
            temp_I += (cluster[key] / total) * math.log( (total * cluster[key]) / (cluster_count * label_counter[key]) )
        I += temp_I
        H_O += (cluster_count / total) * math.log( (cluster_count / total) )

        # Compute "Rand Index"
        # positives: pairs of documents clustered together
        total_positives += float(comb(cluster_count, 2))
        for key in cluster.keys():
            if cluster[key] > 1:
                true_positives += float(comb(cluster[key], 2))
        # negatives: pairs of documents not clustered together
        for cluster_label2, cluster2 in cluster_counter[i+1:]:
            total_negatives += cluster_count * sum(cluster2.values())
            for key in cluster.keys():
                for key2 in cluster2.keys():
                    if key != key2:
                        true_negatives += cluster[key] * cluster2[key2]
    H_O = -H_O

    H_C = 0
    for value in label_counter.values():
        H_C += (value / total) * math.log((value / total))
    H_C = -H_C

    # Return cluster counters, unestimated count, purity, nmi, accuracy
    return cluster_counter, unestimated, purity_count / total, I / ((H_O + H_C) / 2), (true_positives + true_negatives) / (total_positives + total_negatives)

def main():
    items = []
    for estimated_label in glob.glob(os.path.join(ESTIMATED_LABEL_DIR, '*')):
        label = os.path.basename(estimated_label)
        if label in ['templates', 'labeled']:
            continue

        items.extend([(label, ele) for ele in glob.glob(os.path.join(estimated_label, '*'))])

    global score
    cluster_counter, unestimated, purity, nmi, accuracy = score(items)

    for cluster_label, counter in cluster_counter:
        cluster_count = sum(counter.values())
        most_common = counter.most_common(1)[0]
        score = most_common[1] / cluster_count
        print 'Score for label', cluster_label, 'with', cluster_count, 'examples is', score

    print 'Unestimated:', unestimated
    print
    print 'Purity', purity
    print 'NMI', nmi
    print 'Accuracy', accuracy

if __name__ == '__main__':
    main()
