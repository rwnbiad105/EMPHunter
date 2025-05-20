import json, os, pickle
import numpy as np
from gensim.models import FastText
from gensim.models import Word2Vec
import datetime
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
from SequenceGenerator.basedata_samples_parser import TrainParser as BatchParser


def load_model(we_train_batch_name):
    js_wv_model = FastText.load(
        os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name, "js_we.model"))
    cmd_wv_model = FastText.load(
        os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name, "cmd_we.model"))
    return js_wv_model, cmd_wv_model


def bash_parser(preinstall_dict, postinstall_dict, pkg_name_2_file_path_dict,
                batch_name, pkg_id_2_pkg_name_dict, we_train_batch_name, is_canonical):
    #     cmd_seq_dict = 仅包含cmd的脚本内容
    #     js_seq_dict = 包含调用js的脚本内容
    def script_field_analyser(my_parser, field_dict, is_canonical, we_train_batch_name):
        if is_canonical:
            for key, value in field_dict.items():
                root_node = my_parser.parse(value, 'bash')
                my_parser.traverse_bash_ast(root_node, key, we_train_batch_name)
        else:
            for key, value in field_dict.items():
                root_node = my_parser.parse(value, 'bash')
                my_parser.traverse_bash_ast(root_node, key)
        return my_parser.cmd_seq_dict, my_parser.js_seq_dict

    cmd_seq_dict = {}
    js_seq_dict = {}
    if is_canonical:
        js_wv_model, cmd_wv_model = load_model(we_train_batch_name)
        my_parser = BatchParser(is_canonical, cmd_seq_dict, js_seq_dict, pkg_name_2_file_path_dict, batch_name,
                                pkg_id_2_pkg_name_dict)
        my_parser.js_wv_model = js_wv_model
        my_parser.cmd_wv_model = cmd_wv_model
    else:
        my_parser = BatchParser(is_canonical, cmd_seq_dict, js_seq_dict, pkg_name_2_file_path_dict, batch_name,
                                           pkg_id_2_pkg_name_dict)
    script_field_analyser(my_parser, preinstall_dict, is_canonical, we_train_batch_name)
    cmd_seq_dict, js_seq_dict = script_field_analyser(my_parser, postinstall_dict, is_canonical, we_train_batch_name)

    return cmd_seq_dict, js_seq_dict, my_parser.pkg_name_2_file_path_dict, my_parser.pkg_id_2_pkg_name_dict


def install_json_parser(batch_name):
    result_dir = os.path.join(CURRENT_DIR, "../DataShare/Result")
    unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/unpacked", batch_name)
    package_dir = os.path.join(CURRENT_DIR, "../../Packages/packages", batch_name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(os.path.join(result_dir, batch_name)):
        os.mkdir(os.path.join(result_dir, batch_name))
    if not os.path.exists(os.path.join(result_dir, batch_name, "json")):
        os.mkdir(os.path.join(result_dir, batch_name, "json"))

    pkg_list = os.listdir(unpacked_dir)
    downloaded_name_list = []
    preinstall_dict = {}
    postinstall_dict = {}
    downloaded_failed_list = []
    pkg_name_2_file_path_dict = {}
    pkg_id_2_pkg_name_dict = {}
    pkg_id = 0
    for file in pkg_list:
        file_dir = os.path.join(unpacked_dir, file, "package/package.json")
        try:
            with open(file_dir, "r") as f:
                json_data = json.load(f)
                version = json_data["version"]
                pkg_name = json_data["name"] + "-" + version
                downloaded_name_list.append(pkg_name)
                if "scripts" in json_data:
                    scripts = json_data["scripts"]
                    if "preinstall" in scripts:
                        print("preinstall:" + pkg_name)
                        print(scripts["preinstall"] + "\n")
                        pkg_id += 1
                        preinstall_dict[pkg_id] = scripts["preinstall"]
                        pkg_id_2_pkg_name_dict[pkg_id] = pkg_name
                        pkg_name_2_file_path_dict[pkg_name] = os.path.join(unpacked_dir, file)
                    if "postinstall" in scripts:
                        print("postinstall:" + pkg_name)
                        print(scripts["postinstall"] + "\n")
                        pkg_id += 1
                        pkg_id_2_pkg_name_dict[pkg_id] = pkg_name
                        postinstall_dict[pkg_id] = scripts["postinstall"]
                        pkg_name_2_file_path_dict[pkg_name] = os.path.join(unpacked_dir, file)
        except:
            # print("fail to open: " + file_dir + "\n")
            pass
    try:
        with open(os.path.join(package_dir, batch_name + "_names.txt"), "r") as f:
            all_name_list = f.readlines()
    except:
        all_name_list = []

    if all_name_list != []:
        for name in all_name_list:
            if name.strip() not in downloaded_name_list:
                print("fail to download: " + name)
                downloaded_failed_list.append(name.strip())

    return preinstall_dict, postinstall_dict, downloaded_failed_list, unpacked_dir, \
           pkg_name_2_file_path_dict, pkg_id_2_pkg_name_dict


def remove_dup(js_seq_dict, cmd_seq_dict):
    def process_seq_dict(seq_dict, dup_dict):
        tmp_dict = {}
        no_dup_dict = {}
        for pkg_id, seq in seq_dict.items():
            if seq.strip() == "":
                continue
            if seq in tmp_dict.keys():
                if tmp_dict[seq] not in dup_dict.keys():
                    dup_dict[tmp_dict[seq]] = [pkg_id]
                else:
                    dup_dict[tmp_dict[seq]].append(pkg_id)
            else:
                tmp_dict[seq] = pkg_id
                no_dup_dict[pkg_id] = seq

        return no_dup_dict, dup_dict

    js_seq_dict, js_dup_dict = process_seq_dict(js_seq_dict, {})
    cmd_seq_dict, cmd_dup_dict = process_seq_dict(cmd_seq_dict, {})

    js_seq_dict = dict(sorted(js_seq_dict.items(), key=lambda x: x[0]))
    cmd_seq_dict = dict(sorted(cmd_seq_dict.items(), key=lambda x: x[0]))

    return js_seq_dict, cmd_seq_dict, js_dup_dict, cmd_dup_dict


def remove_api(seq):
    with open(os.path.join(CURRENT_DIR, "need_2_remove_apis.txt"), "r") as api_fp:
        target_api_list = api_fp.readlines()
    for target_api in target_api_list:
        seq_list = seq.split(" ")
        while target_api.strip() in seq_list:
            seq_list.remove(target_api.strip())
            seq = " ".join(seq_list)
    return seq


def install_file_parser_main(result_dir, batch_name, we_train_batch_name, is_canonical=True, is_original=False):
    preinstall_dict, postinstall_dict, downloaded_failed_list, unpacked_dir, \
        pkg_name_2_file_path_dict, pkg_id_2_pkg_name_dict = install_json_parser(batch_name)

    print("###################################### start to parser bash ######################################")
    if is_original:
        cmd_seq_dict = {**preinstall_dict, **postinstall_dict}
        js_seq_dict = {}
    else:
        cmd_seq_dict, js_seq_dict, pkg_name_2_file_path_dict, pkg_id_2_pkg_name_dict = bash_parser(preinstall_dict, postinstall_dict, pkg_name_2_file_path_dict,
                                            batch_name, pkg_id_2_pkg_name_dict, we_train_batch_name, is_canonical)

    print("############### cmd_seq ###############")
    for k, v in cmd_seq_dict.items():
        print(pkg_id_2_pkg_name_dict[k] + ": ")
        print(v)
        print()
    print("############### js_seq ###############")
    for k, v in js_seq_dict.items():
        print(pkg_id_2_pkg_name_dict[k] + ": ")
        print(v)
        print()

    js_seq_dict, cmd_seq_dict, js_dup_dict, cmd_dup_dict = remove_dup(js_seq_dict, cmd_seq_dict)

    print("############### js_dup ###############")
    for k, v in js_dup_dict.items():
        if len(v) > 1:
            print(k, ": ")
            for i in v:
                print(pkg_id_2_pkg_name_dict[i])

    print("############### cmd_dup ###############")
    for k, v in cmd_dup_dict.items():
        if len(v) > 1:
            print(k,": ")
            for i in v:
                print(pkg_id_2_pkg_name_dict[i])

    json_str_cmd_seq = json.dumps(cmd_seq_dict, indent=4)
    json_str_js_seq = json.dumps(js_seq_dict, indent=4)
    file_name_cmd_seq = os.path.join(result_dir, batch_name, "raw_cmd_seq.json")
    file_name_js_seq = os.path.join(result_dir, batch_name, "raw_js_seq.json")

    with open(file_name_js_seq, 'w+') as json_file:
        json_file.write(json_str_js_seq)
    with open(file_name_cmd_seq, 'w+') as json_file:
        json_file.write(json_str_cmd_seq)

    file_name_id2pkg = os.path.join(result_dir, batch_name, "id2pkg.json")
    with open(file_name_id2pkg, "w+") as f:
        json_str = json.dumps(pkg_id_2_pkg_name_dict, indent=4)
        f.write(json_str)