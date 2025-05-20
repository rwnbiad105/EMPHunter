import json
import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_samples(batch_name, result_dir):
    # 用来准备每个批次的样本初始化
    file_name_cmd_seq = os.path.join(result_dir, batch_name, "raw_cmd_seq.json")
    file_name_js_seq = os.path.join(result_dir, batch_name, "raw_js_seq.json")

    with open(file_name_js_seq, 'r') as json_file:
        js_seq_dict = json.load(json_file)
    with open(file_name_cmd_seq, 'r') as json_file:
        cmd_seq_dict = json.load(json_file)

    with open(os.path.join(result_dir, batch_name, "labeled_cmd_seq.txt"), "w+") as f:
        for pkg_id, seq in cmd_seq_dict.items():
            f.write("__label__" + str(pkg_id) + " " + seq + "\n")

    with open(os.path.join(result_dir, batch_name, "labeled_js_seq.txt"), "w+") as f:
        for pkg_id, seq in js_seq_dict.items():
            f.write("__label__" + str(pkg_id) + " " + seq + "\n")


def prepare_train_samples(trian_batch_name, json_name):
    # 用来准备训练样本的初始化
    train_sample_dir = os.path.join(CURRENT_DIR, "../../DataShare/Model/SeqEmbedding", trian_batch_name)
    file_name_cmd_seq = os.path.join(train_sample_dir, json_name + ".json")

    with open(file_name_cmd_seq, 'r') as json_file:
        cmd_seq_dict = json.load(json_file)

    cmd_samples_list = []
    count = 0
    for pkg_name, seq in cmd_seq_dict.items():
        count += 1
        cmd_samples_list.append((count, pkg_name, seq))

    with open(os.path.join(train_sample_dir, json_name + "_labeled.txt"), "w+") as f:
        for elm in cmd_samples_list:
            if len(elm[2]) and type(elm[2]) == type("string"):
                f.write("__label__" + str(elm[0]) + " " + elm[2] + "\n")


def prepare_basedata_samples(input_file_name, output_file_name, benign_sample_dir):
    # 用来准备benign集合样本的初始化
    file_name_cmd_seq = os.path.join(benign_sample_dir, input_file_name)

    with open(file_name_cmd_seq, 'r') as json_file:
        cmd_seq_dict = json.load(json_file)

    cmd_samples_list = []
    count = 0
    for pkg_name, seq in cmd_seq_dict.items():
        count += 1
        cmd_samples_list.append((count, pkg_name, seq))

    with open(os.path.join(benign_sample_dir, output_file_name), "w+") as f:
        for elm in cmd_samples_list:
            f.write("__label__" + str(elm[0]) + " " + elm[2] + "\n")