import json, os
from scipy import spatial
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_matrix(config, benign_betch_name):

    ret = []

    benign_matrix_json_name = config[0]
    benign_vec_file_name = config[1]
    target_vec_data_list = config[2]

    # read benign matrix
    # with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Benign", benign_betch_name, benign_matrix_json_name),
    #           "rb") as benign_matrix_json_fp:
    #     benign_dist_matrix = pickle.load(benign_matrix_json_fp)
    # print("size of " + benign_matrix_json_name + ": " + str(len(benign_dist_matrix)))

    # read benign vec
    benign_sample_path = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Benign",
                                      benign_betch_name, benign_vec_file_name)
    with open(benign_sample_path, "r") as benign_fp:
        vec_lines = benign_fp.readlines()
    benign_vec_data_list = read_vec(vec_lines, False)
    print("size of " + benign_matrix_json_name + ": " + str(len(benign_vec_data_list)))
    print("size of target_vec_list : " + str(len(target_vec_data_list)))
    all_vec_list = np.concatenate((benign_vec_data_list, target_vec_data_list))
    final_matrix = cosine_distances(all_vec_list)

    # # compute target matrix
    # target_dist_matrix = []
    # for vec1 in target_vec_data_list:
    #     dist_list = []
    #     for vec2 in target_vec_data_list:
    #         cos_angle = spatial.distance.cosine(vec1, vec2)
    #         dist_list.append(cos_angle)
    #     target_dist_matrix.append(dist_list)
    # print("size of " + benign_vec_file_name + ": " + str(len(target_dist_matrix)))
    #
    # # compose final matrix
    # final_matrix = []
    #
    # index = 0
    # for benign_vec in benign_vec_data_list:
    #     if not np.any(benign_vec):
    #         continue
    #     temp_dist_list = []
    #     for target_vec in target_vec_data_list:
    #         cos_angle = spatial.distance.cosine(benign_vec, target_vec)
    #         temp_dist_list.append(cos_angle)
    #     final_matrix.append(np.concatenate((benign_dist_matrix[index], temp_dist_list), axis=0).tolist())
    #     index += 1
    # ret.append(final_matrix)
    #
    # index = 0
    # for target_vec in target_vec_data_list:
    #     temp_dist_list = []
    #     for benign_vec in benign_vec_data_list:
    #         if not np.any(benign_vec):
    #             continue
    #         cos_angle = spatial.distance.cosine(target_vec, benign_vec)
    #         temp_dist_list.append(cos_angle)
    #     final_matrix.append(np.concatenate((temp_dist_list, target_dist_matrix[index]), axis=0).tolist())
    #     index += 1

    print("size of final matrix: " + str(len(final_matrix)))
    ret.append(final_matrix)
    return ret


def compute_benign_matrix(benign_betch_name, mode):
    # mode == "js" or "cmd"
    if mode == "js":
        vec_file_name = "js_seq_vec.vec"
        matrix_json = "js_benign_matrix.pkl"
    elif mode == "cmd":
        vec_file_name = "cmd_seq_vec.vec"
        matrix_json = "cmd_benign_matrix.pkl"
    benign_sample_path = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Benign", benign_betch_name)
    with open(os.path.join(benign_sample_path, vec_file_name), "r") as benign_fp:
        vec_lines = benign_fp.readlines()
    vec_list_data = read_vec(vec_lines, False)
    # dist_matrix = []
    # for vec1 in vec_list_data:
    #     dist_list = []
    #     for vec2 in vec_list_data:
    #         cosangle = spatial.distance.cosine(vec1, vec2)
    #         dist_list.append(cosangle)
    #     dist_matrix.append(dist_list)

    dist_matrix = cosine_distances(vec_list_data)

    with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Benign", benign_betch_name, matrix_json),
              "wb+") as matrix_json_fp:
        pickle.dump(dist_matrix, matrix_json_fp)


def read_vec(vec_lines, is_contain_zero=True):
    vec_list = []
    for line in vec_lines[:]:
        # line = " ".join(line.split(" ")[1:])
        temp = []
        for i in line.split(" ")[:]:  # for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(float(i))
        if not is_contain_zero and all(item == 0 for item in temp):
            continue
        vec_list.append(temp)

    data = np.array(vec_list)
    return data


def prepare_matrix_main(batch_name, benign_betch_name, is_orginal=False):

    js_vec_dir = os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name)
    cmd_vec_dir = os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name)

    print("\nstart collecting vectors:")
    if not is_orginal:
        with open(os.path.join(js_vec_dir, "js_seq_vec.vec"), "r") as vec_file:
            js_lines = vec_file.readlines()
    with open(os.path.join(cmd_vec_dir, "cmd_seq_vec.vec"), "r") as vec_file:
        cmd_lines = vec_file.readlines()
    js_vec_data = []
    if not is_orginal:
        js_vec_data = read_vec(js_lines, True)
    cmd_vec_data = read_vec(cmd_lines, True)

    print("\nstart computing js matrix:")
    config_list = ["js_benign_matrix.pkl", "js_seq_vec.vec", js_vec_data]
    js_final_matrix_list = [[]]
    if not is_orginal:
        js_final_matrix_list = compute_matrix(config_list, benign_betch_name)
    # with open(os.path.join(js_vec_dir, "js_matrix.pkl"), "wb+") as fp:
    #     pickle.dump(np.array(js_final_matrix_list[0]), fp)

    print("\nstart computing cmd matrix:")
    config_list = ["cmd_benign_matrix.pkl", "cmd_seq_vec.vec", cmd_vec_data]
    final_matrix_list = compute_matrix(config_list, benign_betch_name)
    # with open(os.path.join(cmd_vec_dir, "cmd_matrix.pkl"), "wb+") as fp:
    #     pickle.dump(np.array(final_matrix_list[0]), fp)

    return js_final_matrix_list[0], final_matrix_list[0]


if __name__ == "__main__":
    batch_name = "20240203-2"
    benign_betch_name = "20240224-1"
    min_cluster_size = 2
    min_samples = 10

    # compute_benign_matrix(benign_betch_name, "js")
    # compute_benign_matrix(benign_betch_name, "cmd")
    prepare_matrix_main(batch_name, benign_betch_name)