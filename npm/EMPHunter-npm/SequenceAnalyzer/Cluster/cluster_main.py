import json

import hdbscan
import os
import pickle
from scipy import spatial
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from SequenceAnalyzer.Initialize.compute_distant import compute_matrix, prepare_matrix_main
from sklearn.cluster import HDBSCAN
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_vec(vec_lines):
    vec_list = []
    for line in vec_lines[:]:
        # line = " ".join(line.split(" ")[1:])
        temp = []
        for i in line.split(" ")[:]:    # for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(float(i))
        vec_list.append(temp)
    data = np.array(vec_list)
    return data
    # return vec_list


def do_cluster(matrix, min_cluster_size, min_samples):
    # all_vec_list = np.concatenate((benign_vec_list_data, vec_data), axis=0)
    # cluster = hdbscan.HDBSCAN(min_cluster_size, min_samples, cluster_selection_epsilon=1,
    #                           gen_min_span_tree=True, metric='cosine')
    # cluster_result = cluster.fit(all_vec_list)
    print("the size of the matrix: (" + str(len(matrix)) + "," + str(len(matrix[0])) + ")")
    matrix = np.array(matrix)

    # cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
    #                           gen_min_span_tree=True, metric='precomputed')
    cluster = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                      metric='precomputed')
    cluster_result = cluster.fit(matrix)
    j = 0
    result_noisy_list = []
    result_good_list = []

    # HDBSCAN output
    label_count_dict = {}
    for sat, sat2 in zip(cluster.labels_, cluster_result.probabilities_):
        j = j + 1
        if sat == -1:
            result_noisy_list.append(j)
        # elif sat2 < 0.5:
        #     result_bad_list.append(target_dict[j])
        else:
            result_good_list.append(j)
        if sat not in label_count_dict.keys():
            label_count_dict[sat] = 1
        else:
            label_count_dict[sat] += 1

    result_list = []
    result_list.append(result_good_list)
    print("good list:")
    print(result_good_list)

    # result_list.append(result_bad_list)
    result_list.append(result_noisy_list)
    print("noisy list:")
    print(result_noisy_list)

    result_list.append(cluster.labels_)
    print("label list:")
    print(cluster.labels_)
    print(len(cluster.labels_))
    print(sorted(label_count_dict.items(), key=lambda x:x[1]))

    result_list.append(cluster.probabilities_)
    print("probabilities list:")
    print(cluster.probabilities_)

    return result_list


def hdbscan_cluster_main(batch_name, benign_batch_name, min_cluster_size_js, min_samples_js, min_cluster_size_cmd, min_samples_cmd, is_original):

    js_vec_dir = os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name)
    cmd_vec_dir = os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name)
    js_result_list = []
    cmd_result_list = []

    print("\n#### start collecting matrix: ####")
    # with open(os.path.join(js_vec_dir, "js_matrix.pkl"), "rb") as matrix_fp:
    #     js_matrix = pickle.load(matrix_fp)
    # with open(os.path.join(cmd_vec_dir, "cmd_matrix.pkl"), "rb") as matrix_fp:
    #     cmd_matrix = pickle.load(matrix_fp)
    js_matrix, cmd_matrix = prepare_matrix_main(batch_name, benign_batch_name, is_original)


    print("\n#### start clustering: ####")
    print("################ start to cluster js seq ################")
    if not is_original:
        js_result_list = do_cluster(js_matrix, min_cluster_size_js, min_samples_js)

    print("################ start to cluster cmd seq ################")
    cmd_result_list = do_cluster(cmd_matrix, min_cluster_size_cmd, min_samples_cmd)

    return js_result_list, cmd_result_list
    # return [], cmd_result_list


if __name__ == "__main__":
    batch_name = "20240220-4"
    min_cluster_size = 2
    min_samples = 10

    hdbscan_cluster_main(batch_name, min_cluster_size, min_samples)