import os
import time

from SequenceAnalyzer.Initialize.compute_distant import prepare_matrix_main, compute_benign_matrix
from SequenceAnalyzer.Analysis.embedding import embedding_basedata, embedding_main, mult_process_embedding_basedata
from SequenceAnalyzer.Cluster.cluster_main import hdbscan_cluster_main
from SequenceAnalyzer.Ranking.rank_by_centroid import ranker

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def benign_preparation(benign_batch_name, train_batch_name):
    mult_process_embedding_basedata(benign_batch_name, train_batch_name, "Benign", 16)
    compute_benign_matrix(benign_batch_name, "js")
    compute_benign_matrix(benign_batch_name, "cmd")


def malicious_preparation(malicious_batch_name, train_batch_name):
    embedding_basedata(malicious_batch_name, train_batch_name, "Malicious")


def batch_preparation(batch_name, result_dir, train_batch_name, benign_batch_name):
    import time
    embedding_main(batch_name, result_dir, train_batch_name)
    time_em = time.time()
    prepare_matrix_main(batch_name, benign_batch_name)
    time_merge = time.time()
    return time_em, time_merge


def batch_detection(batch_name, min_cluster_size_js, min_samples_js, min_cluster_size_cmd, min_samples_cmd,
                    benign_batch_name, js_vec_file_name, cmd_vec_file_name, rank_mode, max_n_cluster,
                    malicious_batch_name, is_original=False):
    js_result_list, cmd_result_list = hdbscan_cluster_main(batch_name, benign_batch_name, min_cluster_size_js, min_samples_js,
                                                           min_cluster_size_cmd, min_samples_cmd, is_original)

    print("################ start to rank js seq ################")
    if not is_original:
        js_ranker = ranker()
        samples_num = js_ranker.get_heart(batch_name, benign_batch_name, js_vec_file_name, js_result_list[2], "labeled_js_seq.txt")
        if rank_mode == "centroid":
            js_ranker.ranking_nearest_cluster_centroid_weight_with_malicious(max_n_cluster, malicious_batch_name, "js_seq_vec.vec")
        elif rank_mode == "min":
            js_ranker.ranking_nearest_cluster_with_malicious(max_n_cluster, 1, malicious_batch_name, "js_seq_vec.vec")
        js_ranker.print_result(samples_num, batch_name, "result_js.txt")
    print("################ start to rank cmd seq ################")
    cmd_ranker = ranker()
    samples_num = cmd_ranker.get_heart(batch_name, benign_batch_name, cmd_vec_file_name, cmd_result_list[2], "labeled_cmd_seq.txt")
    if rank_mode == "centroid":
        cmd_ranker.ranking_nearest_cluster_centroid_weight_with_malicious(max_n_cluster, malicious_batch_name, "cmd_seq_vec.vec")
    elif rank_mode == "min":
        cmd_ranker.ranking_nearest_cluster_with_malicious(max_n_cluster, 1, malicious_batch_name, "cmd_seq_vec.vec")
    cmd_ranker.print_result(samples_num, batch_name, "result_cmd.txt")


def prepare_benign_main(benign_batch_name, train_batch_name):
    benign_preparation(benign_batch_name, train_batch_name)


def prepare_malicious_main(benign_batch_name, train_batch_name):
    malicious_preparation(benign_batch_name, train_batch_name)


def analyse_batch_main(batch_name, benign_batch_name, train_batch_name, min_cluster_size_js, min_samples_js,
                       min_cluster_size_cmd, min_samples_cmd, js_vec_file_name, cmd_vec_file_name, rank_mode,
                       max_n_cluster, malicious_batch_name, is_original):
    result_dir = os.path.join(CURRENT_DIR, "../DataShare/Result")
    time_em = time.time()
    time_merge = time.time()
    time_em, time_merge = batch_preparation(batch_name, result_dir, train_batch_name, benign_batch_name)
    batch_detection(batch_name, min_cluster_size_js, min_samples_js, min_cluster_size_cmd, min_samples_cmd, benign_batch_name,
                    js_vec_file_name, cmd_vec_file_name, rank_mode, max_n_cluster, malicious_batch_name, is_original)
    return time_em, time_merge


if __name__ == "__main__":
    batch_name = "20240221-1"

    benign_batch_name = "20240224-1"

    js_model_name = "20240219-js"
    cmd_model_name = "20240222-cmd"
    min_cluster_size = 2
    min_samples = 2
    js_vec_file_name = "js_seq_vec.vec"
    cmd_vec_file_name = "cmd_seq_vec.vec"
    train_batch_name = "20240219-1"

    analyse_batch_main(batch_name, benign_batch_name, train_batch_name, min_cluster_size, min_samples,
                       js_vec_file_name, cmd_vec_file_name, benign_batch_name)


