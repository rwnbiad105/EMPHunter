# from PackageSourceCollector.download_main import download_new_updated_npm_pkgs
from SequenceGenerator.batch_sample_parser import install_file_parser_main
from SequenceGenerator.basedata_samples_parser import train_se_samples_parser_main, benign_samples_parser_main, \
    malicious_samples_parser_main, train_we_samples_parser_main, merge_malicious_samples_main, batch_samples_parser_main
from SequenceGenerator.ablation_sample_parser import benign_ablation_exm_samples_parser_main
from SequenceAnalyzer.train_main import train_all, train_wv_model
from SequenceAnalyzer.analyse_main import analyse_batch_main, prepare_benign_main, prepare_malicious_main
from PackageSourceCollector.pkg_unpacker import unpack_4_train
import datetime
import os
import time
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_main(se_train_batch_name, se_train_sample_dir, we_train_batch_name, we_train_sample_dir, is_canonical, is_original):
    # 解压指定文件夹中的
    # unpack_4_train(se_train_sample_dir)

    # 获取we训练集的seq结果
    train_we_samples_parser_main(we_train_batch_name, we_train_sample_dir, is_original)
    time1 = time.time()
    # 训练canonical we模型
    train_wv_model(we_train_batch_name)
    time2 = time.time()
    # 获取se训练集的seq结果
    train_se_samples_parser_main(se_train_batch_name, se_train_sample_dir, we_train_batch_name, is_canonical, is_original)
    time3 = time.time()
    # 训练两个se模型
    train_all(se_train_batch_name)
    return time1, time2, time3


def ablation_benign_main(benign_batch_name, un_benign_batch_name, se_train_batch_name, un_se_train_batch_name,
                         benign_sampls_dir, benign_limit, we_train_batch_name):
    # 获取benign集的seq并随机挑选出benign_limit个
    benign_ablation_exm_samples_parser_main(benign_batch_name, un_benign_batch_name, benign_sampls_dir, benign_limit,
                                            we_train_batch_name)
    # embed benign seq 并计算距离矩阵
    prepare_benign_main(benign_batch_name, se_train_batch_name)
    prepare_benign_main(un_benign_batch_name, un_se_train_batch_name)


def malicious_main(malicious_batch_name, se_train_batch_name, malicious_sh_samples_dir, malicious_js_samples_dir, we_train_batch_name, is_canonical, is_original):
    malicious_samples_parser_main(malicious_batch_name + "-sh", malicious_sh_samples_dir, we_train_batch_name,
                                  is_canonical, is_original)
    malicious_samples_parser_main(malicious_batch_name + "-js", malicious_js_samples_dir, we_train_batch_name,
                                  is_canonical, is_original)
    merge_malicious_samples_main(malicious_batch_name)
    time5 = time.time()
    prepare_malicious_main(malicious_batch_name, se_train_batch_name)
    return time5


def benign_main(benign_batch_name, se_train_batch_name, benign_sampls_dir, benign_limit, we_train_batch_name, is_canonical, is_original):
    # 获取benign集的seq并随机挑选出benign_limit个
    benign_samples_parser_main(benign_batch_name, benign_sampls_dir, benign_limit, we_train_batch_name, is_canonical, is_original)
    time4 = time.time()
    # embed benign seq 并计算距离矩阵
    prepare_benign_main(benign_batch_name, se_train_batch_name)
    return time4

    
def download_main(download_limit):
    date = str(datetime.date.today())
    date = "".join(date.split("-"))
    # batch_name = download_new_updated_npm_pkgs(date, download_limit)

    return batch_name


def analyse_main(batch_name, target_dir, benign_batch_name, se_train_batch_name, min_cluster_size_js, min_samples_js,
                 min_cluster_size_cmd, min_samples_cmd, rank_mode, max_n_cluster, malicious_batch_name,
                 we_train_batch_name, is_canonical, is_original):
    # 获取batch的seq
    batch_samples_parser_main(batch_name, target_dir, we_train_batch_name, is_canonical, is_original)
    time_pre = time.time()
    # embed，计算距离矩阵，进行排名
    js_vec_file_name = "js_seq_vec.vec"
    cmd_vec_file_name = "cmd_seq_vec.vec"
    time_em, time_merge = analyse_batch_main(batch_name, benign_batch_name, se_train_batch_name, min_cluster_size_js,
                                             min_samples_js, min_cluster_size_cmd, min_samples_cmd, js_vec_file_name,
                                             cmd_vec_file_name, rank_mode, max_n_cluster, malicious_batch_name, is_original)
    return time_pre, time_em, time_merge


if __name__ == "__main__":
    start_time = time.time()
    we_train_batch_name = "20240711-1"

    is_canonical = True # for ablation study, don't change.
    is_original = False # for ablation study, don't change.

    benign_sampls_dir = os.path.join(CURRENT_DIR, "../Packages/benign_sample/20240711-1")
    malicious_sh_samples_dir = os.path.join(CURRENT_DIR, "../Packages/malicious_sample", "20240805-sh")
    malicious_js_samples_dir = os.path.join(CURRENT_DIR, "../Packages/malicious_sample", "20240805-js")
    se_train_sample_dir = os.path.join(CURRENT_DIR, "../Packages/train_sample/20240702-1")
    we_train_sample_dir = os.path.join(CURRENT_DIR, "../Packages/train_sample/20240702-1") 
    batch_dir = os.path.join(CURRENT_DIR, "../Packages/unpacked", "canonical-20240623-a")

    if is_canonical:
        se_train_batch_name = "20240711-1-canonical"
        benign_batch_name = "20240711-2-canonical" 
        malicious_batch_name = "20240812-1-canonical" 
        batch_name = "0812-canonical-20240623-a"

    else:   # for ablation study, don't change.
        se_train_batch_name = "20240711-1"   # uncanonical
        benign_batch_name = "20240812-1"  # uncanonical
        malicious_batch_name = "20240812-1-uncanonical"  # uncanonical
        batch_name = "0812-uncanonical-20240623-a"
    if is_original:
        we_train_batch_name = "20240711-1-NA"  # "20240702-1"
        se_train_batch_name = "20240711-1-NA"   # uncanonical
        benign_batch_name = "20240812-1-NA"  # uncanonical
        malicious_batch_name = "20240812-1-NA"  # uncanonical
        batch_name = "0812-NA-20240623-a"

    mode = "ft"
    benign_limit = 10000
    download_limit = 500

    max_n_cluster = 3
    min_cluster_size_js = 2
    min_samples_js = 5
    min_cluster_size_cmd = 2
    min_samples_cmd = 25
    rank_mode = "min"

    # time1, time2, time3 = train_main(se_train_batch_name, se_train_sample_dir, we_train_batch_name, we_train_sample_dir, is_canonical, is_original)
    end_time1 = time.time()
    # time4 = benign_main(benign_batch_name, se_train_batch_name, benign_sampls_dir, benign_limit, we_train_batch_name, is_canonical, is_original)
    end_time2 = time.time()
    # time5 = malicious_main(malicious_batch_name, se_train_batch_name, malicious_sh_samples_dir, malicious_js_samples_dir, we_train_batch_name, is_canonical, is_original)
    end_time3 = time.time()

    # batch_name = download_main(download_limit)

    time_pre, time_em, time_merge = analyse_main(batch_name, batch_dir, benign_batch_name, se_train_batch_name, min_cluster_size_js, min_samples_js,
                 min_cluster_size_cmd, min_samples_cmd, rank_mode, max_n_cluster, malicious_batch_name,
                 we_train_batch_name, is_canonical, is_original)
    print("done")
    # 结束计时
    end_time4 = time.time()
    # 计算时间差
    # execution_time = time1 - start_time
    # print("we preprocess时间: ", execution_time, "秒")
    # execution_time = time2 - time1
    # print("we训练时间: ", execution_time, "秒")
    # execution_time = time3 - time2
    # print("se preprocess时间: ", execution_time, "秒")
    # execution_time = end_time1 - time3
    # print("se训练时间: ", execution_time, "秒")
    # execution_time = time4 - end_time1
    # print("benign集preprocess代码执行时间: ", execution_time, "秒")
    # execution_time = end_time2 - time4
    # print("benign集嵌入代码执行时间: ", execution_time, "秒")
    # execution_time = time5 - end_time2
    # print("malicious preprocess time cost: ", execution_time, "秒")
    # execution_time = end_time3 - time5
    # print("malicious集嵌入代码执行时间: ", execution_time, "秒")
    # execution_time = time_pre - end_time3
    # print("target集preprocess代码执行时间: ", execution_time, "秒")
    # execution_time = time_em - time_pre
    # print("target集embedding代码执行时间: ", execution_time, "秒")
    # execution_time = time_merge - time_em
    # print("target集merging代码执行时间: ", execution_time, "秒")
    # execution_time = end_time4 - time_merge
    # print("target集cluster&ranking代码执行时间: ", execution_time, "秒")

    with open(os.path.join(CURRENT_DIR, "DataShare/Result/", batch_name, "config.txt"), "w+") as fp:
        for i in [we_train_batch_name, is_canonical, se_train_batch_name, benign_batch_name, malicious_batch_name,
                  batch_name, mode, benign_limit, download_limit, max_n_cluster, min_cluster_size_js, min_samples_js,
                  min_cluster_size_cmd, min_samples_cmd, rank_mode, benign_sampls_dir, malicious_sh_samples_dir, malicious_js_samples_dir,
                  se_train_sample_dir, we_train_sample_dir]:
            fp.write(str(i) + "\n")