import json
import os
import math
from builtins import print, dict

import numpy as np
from scipy.special import expit
from scipy import spatial
import heapq
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/DataShare/ApiSeq_and_Result/"


class ranker():
    def __init__(self):
        self.vec_dict = {}
        self.clust_heart_dict = {}
        self.clust_num_dict = {}
        self.rank_dict = {}
        self.dist_dict = {}
        self.var_dict = {}
        self.weight_dict = {}
        self.weight_dict_2 = {}
        self.label_list = []
        self.all_sample_lable_list = []
        self.malicious_vec_dict = {}
        self.benign_len = 0
        self.clust_dist_list_dict = {}
        self.clust_min_dist_dict = {}
        self.target_seq_lines = []

    def get_heart(self, batch_name, benign_batch_name, vec_file_name, label_list, seq_file_name):
        self.all_sample_lable_list = label_list
        with open(os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name, vec_file_name), "r") as fp:
            target_vec_lines = fp.readlines()
            vec_len = len(target_vec_lines)

        with open(os.path.join(CURRENT_DIR, "../../DataShare/Result", batch_name, seq_file_name), "r") as fp:
            self.target_seq_lines = fp.readlines()

        with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Benign", benign_batch_name, vec_file_name), "r") as fp:
            benign_vec_lines = fp.readlines()
            # remove 0-vec
            benign_vec_lines_no_zero = []
            for line in benign_vec_lines:
                tmp_list = line.strip().split()
                if all(item == "0" for item in tmp_list):
                    continue
                benign_vec_lines_no_zero.append(line)
            self.benign_len = len(benign_vec_lines_no_zero)

        print("\n#### start get centroid heart: ####")
        print(label_list[-1 * vec_len:])

        all_vec_lines = np.concatenate((benign_vec_lines_no_zero, target_vec_lines), axis=0).tolist()
        i = 0
        # 读取所有的target句向量
        for line in all_vec_lines:
            i = i + 1
            vec_list = []
            line = line.split(" ")
            for num in line:
                if num.strip() == "":
                    continue
                vec_list.append(float(num))
            self.vec_dict[i] = vec_list

        # 求质心， 求簇内向量和
        for label, vec in zip(label_list, self.vec_dict.values()):
            if label == -1:
                continue
            if label in self.clust_num_dict.keys():
                self.clust_num_dict[label] += 1
            else:
                self.clust_num_dict[label] = 1

            if label in self.clust_heart_dict.keys():
                current_vec = self.clust_heart_dict[label]
            else:
                self.clust_heart_dict[label] = vec
                continue

            new_vec = []
            for a, b in zip(current_vec, vec):
                new_vec.append(a + b)
            self.clust_heart_dict[label] = new_vec
        print(self.clust_num_dict)

        # 处以簇内的向量个数， 求平均
        for label, num in self.clust_num_dict.items():
            current_vec = self.clust_heart_dict[label]
            new_vec = []
            for i in current_vec:
                new_vec.append(i / num)
            self.clust_heart_dict[label] = new_vec
            self.label_list.append(label)

        # # 算每个向量与每个质心的距离列表
        # for label_num, vec in self.vec_dict.items():
        #     dist_list = []
        #     for label, heart_vec in self.clust_heart_dict.items():
        #         cos_dist = spatial.distance.cosine(vec, heart_vec)
        #         dist_list.append(cos_dist)
        #     self.dist_dict[label_num] = dist_list

        # for i in self.clust_heart_dict.values():
        #     print(i)
        return vec_len

    def print_result(self, samples_num, batch_name, output_fname):
        final_result_dict = {}
        for k, v in self.rank_dict.items():
            if k > self.benign_len and self.all_sample_lable_list[k-1] == -1:
                final_result_dict[k - self.benign_len] = v
        final_result_dict = self.rank_dict
        sorted_list = sorted(final_result_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        with open(os.path.join(CURRENT_DIR, "../../DataShare/Result/", batch_name, output_fname), "w+") as fp:
            count = 0
            for k, v in dict(sorted_list).items():
                if self.all_sample_lable_list[k + self.benign_len - 1] == -1:
                    count += 1
                print(str(count), "label:", self.all_sample_lable_list[k + self.benign_len - 1], str(k) + " " + str(v))
                fp.write(str(count) + " label: " + str(self.all_sample_lable_list[k + self.benign_len - 1]) + " id:" + str(k) + " score:" + str(v) + "\n")

    def ranking_nearest_cluster_centroid_weight_with_malicious(self, closest_cluster_n, malicious_batch_name, malicious_vec_file):
        # get malicious dist dict
        with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Malicious/",
                               malicious_batch_name, malicious_vec_file)) as fp:
            malicious_vec_lines = fp.readlines()
            i = 0
            for line in malicious_vec_lines:
                i = i + 1
                vec_list = []
                line = line.split(" ")
                for num in line:
                    if num.strip() == "":
                        continue
                    vec_list.append(float(num))
                np_vec = np.array(vec_list)
                if np.count_nonzero(np_vec) != 0:
                    self.malicious_vec_dict[i] = vec_list

        # 计算每一个样本距离最近的恶意样本的距离
        malicious_dist_dict = {}
        for label_num, vec in self.vec_dict.items():
            if label_num <= self.benign_len:
                continue
            smallest_dist = 100000
            for m_label, m_vec in self.malicious_vec_dict.items():
                m_cos_dist = spatial.distance.cosine(vec, m_vec)
                if m_cos_dist < smallest_dist:
                    smallest_dist = m_cos_dist
            malicious_dist_dict[label_num - self.benign_len] = smallest_dist
        print("malicious_dist_dict:")
        print(malicious_dist_dict)

        # 算每个向量与每个质心的距离列表
        for label_num, vec in self.vec_dict.items():
            if label_num <= self.benign_len:
                continue
            dist_dic = {}
            for label, heart_vec in self.clust_heart_dict.items():
                cos_dist = spatial.distance.cosine(vec, heart_vec)
                dist_dic[label] = cos_dist
            sorted_dict = sorted(dist_dic.items(), key=lambda x: x[1], reverse=False)
            if closest_cluster_n > len(sorted_dict):
                closest_cluster_n = len(sorted_dict)
            # 只取字典的前closest_cluster_n个
            near_cluster_dist_dict = dict(sorted_dict[:closest_cluster_n])
            self.dist_dict[label_num - self.benign_len] = near_cluster_dist_dict

        # 算每个簇的权重, 簇越大权重越低
        for label, num in self.clust_num_dict.items():
            self.weight_dict[label] = math.log(num, 10)

        # 算向量的rank得分， 分高的向量是我们想要的
        for label_num, dist_dic in self.dist_dict.items():
            line_len = len(self.target_seq_lines[label_num - 1].split(" ")) - 1
            Wb = 0.5 - (1 - expit(line_len - 1))/5
            Wm = 1- Wb

            # score_list = heapq.nsmallest(3, dist_list)
            score = 0
            # print(dist_list)
            for label, i in dist_dic.items():
                score += i # * self.weight_dict[label]
            score = score/closest_cluster_n - malicious_dist_dict[label_num]
            # score = Wb * score / closest_cluster_n - Wm * malicious_dist_dict[label_num]
            self.rank_dict[label_num] = score

        return self.rank_dict

    def ranking_nearest_cluster_with_malicious(self, closest_cluster_n, closest_benign_n, malicious_batch_name, malicious_vec_file):
        # get malicious dist dict
        with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/Malicious/",
                               malicious_batch_name, malicious_vec_file)) as fp:
            malicious_vec_lines = fp.readlines()
            i = 0
            for line in malicious_vec_lines:
                i = i + 1
                vec_list = []
                line = line.split(" ")
                for num in line:
                    if num.strip() == "":
                        continue
                    vec_list.append(float(num))
                np_vec = np.array(vec_list)
                if np.count_nonzero(np_vec) != 0:
                    self.malicious_vec_dict[i] = vec_list

        # 计算每一个样本距离最近的恶意样本的距离
        malicious_dist_dict = {}
        for label_num, vec in self.vec_dict.items():
            if label_num <= self.benign_len:
                continue
            smallest_dist = 100000
            s_m_label = 0
            if np.all(vec):
                for m_label, m_vec in self.malicious_vec_dict.items():
                    m_cos_dist = spatial.distance.cosine(vec, m_vec)
                    if not np.all(m_vec):
                        continue
                    if m_cos_dist == 0:
                        # print(label_num, m_label)
                        pass
                    if m_cos_dist < smallest_dist:
                        smallest_dist = m_cos_dist
                        s_m_label = m_label
            malicious_dist_dict[label_num - self.benign_len] = [smallest_dist, s_m_label]
        print("malicious_dist_dict:")
        print(malicious_dist_dict)

        # 算每个向量与每个簇中最近的closest_benign_n个向量的距离列表
        for label_num, vec in self.vec_dict.items():
            clust_dist_list_dict = {}
            clust_label_list_dict = {}
            clust_min_dist_dict = {}
            if label_num <= self.benign_len:
                continue
            count = 0
            for label, c_vec in zip(self.all_sample_lable_list, self.vec_dict.values()):
                count += 1
                if label == -1 or count > self.benign_len:
                    continue
                current_dist = spatial.distance.cosine(vec, c_vec)

                if not np.all(vec) or not np.all(c_vec):
                    continue
                if current_dist == 0:
                    if count == label_num:
                        continue
                    else:
                        # print(label_num, count)
                        pass
                if label in clust_dist_list_dict.keys():
                    clust_dist_list_dict[label].append(current_dist)
                else:
                    clust_dist_list_dict[label] = [current_dist]
                clust_label_list_dict[label] = count
            # print(label_num)
            # print(clust_label_list_dict)

            for label, dist_list in clust_dist_list_dict.items():
                # if label == 255 and label_num == 10060:
                #     print(dist_list)
                sorted_list = sorted(dist_list, reverse=False)
                sum = 0
                if closest_benign_n > len(sorted_list):
                    c_benign_n = len(sorted_list)
                else:
                    c_benign_n = closest_benign_n
                for dist in sorted_list[:c_benign_n]:
                    sum += dist
                average_dist = sum / c_benign_n
                clust_min_dist_dict[label] = average_dist

            sorted_list = sorted(clust_min_dist_dict.items(), key=lambda x: x[1], reverse=False)

            # 只取字典的前min_n个
            if closest_cluster_n > len(sorted_list):
                temp_closest_cluster_n = len(sorted_list)
            else:
                temp_closest_cluster_n = closest_cluster_n
            near_cluster_dist_list = sorted_list[:temp_closest_cluster_n]
            near_cluster_dist_dict = dict(near_cluster_dist_list)
            self.dist_dict[label_num - self.benign_len] = near_cluster_dist_dict

        # 算每个簇的权重, 簇越大权重越大
        # for label, num in self.clust_num_dict.items():
        #     self.weight_dict[label] = math.log(num, 10)

        # 算向量的rank得分， 分高的向量是我们想要的
        for label_num, dist_dic in self.dist_dict.items():
            line_len = len(self.target_seq_lines[label_num - 1].split(" ")) - 1
            Wb = 0.5 - (1 - expit(line_len - 1))/5
            Wm = 1- Wb
            # print(label_num, Wb)
            # print(dist_dic)

            # score_list = heapq.nsmallest(3, dist_list)
            score = 0
            for label, i in dist_dic.items():
                score += i # * self.weight_dict[label]
            # print(score)
            score = Wb * score/closest_cluster_n - Wm * malicious_dist_dict[label_num][0]
            # score = - Wm * malicious_dist_dict[label_num][0]
            # score = Wb * score / closest_cluster_n
            self.rank_dict[label_num] = score

        return self.rank_dict


if __name__ == "__main__":
    batch_name = "20240220-4"
    js_vec_file_name = "js_seq_vec.vec"
    cmd_vec_file_name = "cmd_seq_vec.vec"
    # with open(os.path.join(CURRENT_DIR, "../../DataShare/")) as fp:
    # label_list
    # ranker = ranker()
    # ranker.get_heart(batch_name, vec_file_name, label_list)