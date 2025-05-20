from SequenceAnalyzer.Cluster.cluster_main import cluster_main_hdbscan, cluster_main_dbscan
from SequenceAnalyzer.Visualization.sentence_embedding_visualize import d2

import os, pickle
import numpy as np

# =====================================================================================
# 本程序会计算各个聚类质心之间的距离, 并且不会覆盖原来的聚类结果
# 不需要其他模型的支持
# =====================================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sentence_embedding_analyse_main_calculate_centroid_without_output(proj_name, negative_proj_name, rank_mode ,
                                                                      method, merge_len):
    ret_list = cluster_main_hdbscan(method, proj_name)
    good = ret_list[0]
    noise = ret_list[1]
    labels = ret_list[2]
    probabilities = ret_list[3]

    if negative_proj_name != "":
        negative_ret_list = cluster_main_hdbscan(method, negative_proj_name)
        negative_good = negative_ret_list[0]
        negative_noise = negative_ret_list[1]
        negative_labels = negative_ret_list[2]
        negative_probabilities = negative_ret_list[3]

    if rank_mode == "positive":
        from SequenceAnalyzer.Ranking.rank_by_centroid_positive_sample_only import ranker
        ranker = ranker()
        # ranker.get_heart(proj_name, labels)
        # rank_dict = ranker.ranking_all_cluster_weight()
        rank_dict = ranker.ranking_top_seq(proj_name, labels)
        rank_list = sorted(rank_dict.items(), key=lambda d: d[1], reverse=True)
    elif rank_mode == "negative":
        from SequenceAnalyzer.Ranking.rank_by_centroid_negative_sample_only import ranker
        ranker = ranker()
        ranker.get_heart(proj_name, negative_proj_name, negative_labels)
        rank_dict = ranker.ranking_by_seq()
        rank_list = sorted(rank_dict.items(), key=lambda d: d[1], reverse=False)
    elif rank_mode == "p&n":
        from SequenceAnalyzer.Ranking.rank_by_centroid_all_sample import ranker
        ranker = ranker()
        ranker.get_heart(proj_name, negative_proj_name, labels)
        # rank_dict, negative_dist_dict = ranker.ranking_by_p_centroid_n_seq()
        rank_dict, negative_dist_dict, positive_dict = ranker.ranking_by_p_seq_one_n_seq(proj_name, labels, merge_len)
        # rank_dict, negative_dist_dict, positive_dict, label_dup_dict = ranker.ranking_by_p_seq_one_n_seq_new(proj_name, labels, merge_len)
        # rank_dict, negative_dist_dict = ranker.ranking_by_p_seq_n_seq(proj_name, labels)
        rank_list = sorted(rank_dict.items(), key=lambda d: d[1], reverse=True)

    # result report
    with open(
            PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
            proj_name + "_" + method + "_target_num_dict.pkl", "rb") as file:
        target_dict = pickle.load(file)
    with open(
            PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
            proj_name + "_" + method + "_label_file_dict.pkl", "rb") as file:
        addr_dict = pickle.load(file)
    with open(
            PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
            proj_name + "_" + method + "_target_seq.txt", "r") as vec_file:
        lines = vec_file.readlines()

    # result output
    rank_label_dist = {}
    if rank_mode == "p&n":
        if os.path.exists(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name +
                  "/seq_dup_dict.pkl"):
            with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name +
                      "/seq_dup_dict.pkl", "rb+") as file:
                seq_dup_dict = pickle.load(file)
                # print("seq_dup_dict")
                # print(seq_dup_dict)
        else:
            seq_dup_dict = {}
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
                  proj_name + "_result4.txt", "w+") as file:
            i = 0
            line_count = 0
            for v in rank_list:
                label = v[0]
                this_dist_list = ranker.dist_dict[label]
                label_dist_dict = {}
                j = 0
                for dist in this_dist_list:
                    label_dist_dict[j] = dist
                    j = j + 1
                i = i + 1

                negative_dist_list = negative_dist_dict[label]
                positive_score = positive_dict[label]

                # print(i, v, labels[label - 1], min(negative_dist_list),
                #       negative_dist_list.index(min(negative_dist_list)) + 1, negative_dist_list)
                # temp_list = [str(i), str(v[0]), str(v[1]), str(labels[label - 1]), str(min(negative_dist_list)),
                #              str(positive_score),
                #              str(negative_dist_list.index(min(negative_dist_list)) + 1), "\n" + str(addr_dict[label]),
                #              str(lines[label - 1]), str(label_dup_dict[lines[label - 1]]) + "\n"]

                if v[0] >= merge_len:
                    if v[0] - 2203 in seq_dup_dict.keys():
                        line_count = seq_dup_dict[v[0] - 2203]
                    else:
                        line_count = 0
                        # print(v[0] - merge_len)
                    temp_list = ["##", str(i), str(line_count), str(v[0]), str(v[1]), str(labels[label - 1]),
                                 str(min(negative_dist_list)),
                                 str(positive_score), str(negative_dist_list.index(min(negative_dist_list)) + 1),
                                 "\n" + str(addr_dict[label]), str(lines[label - 1])]
                    rank_label_dist[v[0]] = i
                    for j in temp_list:
                        file.write(j + " ")
                    file.write("[")
                    negative_dist_list.sort()
                    count = 0
                    for j in negative_dist_list:
                        count += 1
                        file.write(str(j) + " ")
                        if count == 10:
                            break
                    file.write("]\n")

    else:
        if rank_mode == "positive":
            x = 3
        else:
            x = 2
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name
                  + "_result" + str(x) + ".txt", "w+") as positive_file:
            i = 0
            rank_num = 0
            for v in rank_list:
                label = v[0]
                this_dist_list = ranker.dist_dict[label]
                label_dist_dict = {}
                j = 0
                for dist in this_dist_list:
                    label_dist_dict[j] = dist
                    j = j + 1

                i = i + 1
                temp_list = []
                k = 0
                if label >= merge_len:
                    if rank_mode == "positive":
                        dist_rank_list = sorted(label_dist_dict.items(), key=lambda d: d[1], reverse=True)
                    elif rank_mode == "negative":
                        dist_rank_list = sorted(label_dist_dict.items(), key=lambda d: d[1], reverse=False)
                    rank_num += 1
                    rank_label_dist[label] = i

                    # get distance to every clust
                    for j in dist_rank_list:
                        k += 1
                        if k <= 100:
                            temp_list.append((j[0], j[1]))
                    # print(i, v, labels[label - 1], temp_list)
                    temp = [str(rank_num), str(v[0]), str(v[1]), str(labels[label - 1]), str(addr_dict[label]), str(lines[label-1])]
                    for j in temp:
                        positive_file.write(j + " ")
                    # get distance to every clust
                    positive_file.write("[")
                    for k in temp_list:
                        positive_file.write("(" + str(k[0]) + "," + str(k[1]) + ")")
                    positive_file.write("]\n")

    all_heart_vec_list = []
    for vec in ranker.clust_heart_dict.values():
        all_heart_vec_list.append(vec)
    if rank_mode == "positive":
        # d2(all_heart_vec_list, ranker.label_list)
        pass

    new_d = {v: k for k, v in target_dict.items()}
    temp_list = []
    for i in noise:
        label = new_d[i]
        temp_list.append(label)
    noise_list = []
    good_seq_list = []
    i = 0
    for line in lines:
        i = i + 1
        if i in temp_list:
            noise_list.append(line)
        else:
            good_seq_list.append(line)

    clust_num = np.max(labels)
    print(clust_num)
    # 借助锚点来筛选top_rank  20230321 =========================================================
    cmd = "cp " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/PreProcess/false_negative.txt " + PROJECT_DIR + \
          "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/false_negative.txt"
    os.system(cmd)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name +
              "/false_negative.txt", "r") as fp:
        false_negative_list = fp.readline()
        false_negative_num = len(false_negative_list)
        min_false_negative_rank = 20000 + false_negative_num
    for i in range(clust_num + 2):
        clust = i - 1
        j = 0
        clust_item_count = 0
        for item, l, p in zip(lines, labels, probabilities):
            j = j + 1
            if item.strip() == "":
                continue
            if clust == l:
                clust_item_count += 1
                if addr_dict[j].split("/")[-1] in false_negative_list and rank_label_dist[j] < min_false_negative_rank:
                    min_false_negative_rank = rank_label_dist[j]
    # 20230321 end ===========================================================================

    if rank_mode == "p&n":
        with open(
                PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +
                "_ranked_analyse-p&n-result.txt", "w+") as fp:
            for i in range(clust_num + 2):
                clust = i - 1
                j = 0
                clust_item_count = 0
                output = ""
                for item, l, p in zip(lines, labels, probabilities):
                    j = j + 1
                    if item.strip() == "":
                        continue
                    if clust == l:
                        clust_item_count += 1

                        if j in rank_label_dist.keys() and rank_label_dist[j] < min_false_negative_rank:
                            output += str(rank_label_dist[j]) + " " + str(j) + " clust:" + str(l) + " prob:" + str(
                                p) + " seq:" + item + " addr:" + addr_dict[j]
                print("clust:" + str(clust) + "       items:" + str(clust_item_count))
                # print(output)
                fp.write("###############          clust:" + str(clust) + "   items:" + str(
                    clust_item_count) + "         #################\n")
                fp.write(output + "\n\n")
    elif rank_mode == "positive":
        x = 3
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +
                "_ranked_analyse-positive-result.txt", "w+") as positive_file:
            i = 0
            rank_num = 0
            for v in rank_list:
                label = v[0]
                this_dist_list = ranker.dist_dict[label]
                label_dist_dict = {}
                j = 0
                for dist in this_dist_list:
                    label_dist_dict[j] = dist
                    j = j + 1

                i = i + 1
                temp_list = []
                k = 0
                if rank_mode == "positive":
                    dist_rank_list = sorted(label_dist_dict.items(), key=lambda d: d[1], reverse=True)
                    # dist_rank_list = sorted(label_dist_dict.items(), key=lambda d: d[1], reverse=False)
                elif rank_mode == "negative":
                    dist_rank_list = sorted(label_dist_dict.items(), key=lambda d: d[1], reverse=False)
                rank_num += 1
                rank_label_dist[label] = i

                # get distance to every clust
                for j in dist_rank_list:
                    k += 1
                    if k <= 100:
                        temp_list.append((j[0], j[1]))
                # print(i, v, labels[label - 1], temp_list)
                temp = [str(rank_num), str(v[0]), str(v[1]), str(labels[label - 1]), str(addr_dict[label]), str(lines[label-1])]
                for j in temp:
                    positive_file.write(j + " ")
                # get distance to every clust
                positive_file.write("[")
                for k in temp_list:
                    positive_file.write("(" + str(k[0]) + "," + str(k[1]) + ")")
                positive_file.write("]\n")
        # with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_rank_dist_dict.pkl",
        #           "wb+") as file:
        #     pickle.dump(ranker.dist_dict, file)
    #     with open(
    #             PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +
    #             "_ranked_analyse-positive-result.txt", "w+") as fp:
    #         # print("start")
    #         # print(rank_label_dist.keys())
    #         for i in range(clust_num):
    #             clust = i + 1
    #             j = 0
    #             clust_item_count = 0
    #             output = ""
    #             for item, l, p in zip(lines, labels, probabilities):
    #                 j = j + 1
    #                 if item.strip() == "":
    #                     continue
    #                 if clust == l:
    #                     # print(j)
    #                     clust_item_count += 1
    #                     if j in rank_label_dist.keys() and rank_label_dist[j] < min_false_negative_rank:
    #                         output += str(rank_label_dist[j]) + " " + str(j) + " clust:" + str(l) + " prob:" + str(
    #                             p) + " seq:" + item + " addr:" + addr_dict[j]
    #             print("clust:" + str(clust) + "       items:" + str(clust_item_count))
    #             # print(output)
    #             fp.write("###############          clust:" + str(clust) + "   items:" + str(
    #                 clust_item_count) + "         #################\n")
    #             fp.write(output + "\n\n")


def take_top_rank_by_false_negative(merged_proj_name, top_num):
    cmd = "cp " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/PreProcess/false_negative.txt " + PROJECT_DIR + \
          "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/false_negative.txt"
    os.system(cmd)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +
                "_ranked_analyse-result.txt", "w+") as fp:
        lines = fp


if __name__ == "__main__":

    # model_name = "0113_6000-pkg_new_updated_pypi_setup_c.x"   # "0102_13000pkg_ft_d200_e50_rm3"   # gai
    proj_name = "0306_7720-pkg_new_updated_pypi_setup_c.x_merged"
    negative_proj_name = "0304_900-pkg_malicious_setup_c.x"  # "0226_800-pkg_all_malicious_c.x"
    rank_mode = "p&n"  # "positive" or "negative" or "p&n"

    method = "ft"
    merge_len = 1478

    sentence_embedding_analyse_main_calculate_centroid_without_output(proj_name, negative_proj_name, rank_mode, method,
                                                                      merge_len)








