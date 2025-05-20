import json, os, random
from SequenceGenerator.basedata_samples_parser import TrainParser, check_pkg_metadata, \
    load_model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_random_from_json_ablation_exm(benign_batch_name, un_benign_batch_name, benign_limit):
    def select_random_elements_from_json_file_ablation_exm(file_path, un_file_path, benign_limit, new_file_path, un_new_file_path):
        with open(file_path, 'r') as file:
            # 从文件中加载 JSON 数据并解析为字典
            original_dict = json.load(file)

        with open(un_file_path, 'r') as file:
            # 从文件中加载 JSON 数据并解析为字典
            un_original_dict = json.load(file)

        select_len = min(benign_limit, len(original_dict.keys()), len(un_original_dict.keys()))
        keys = list(original_dict.keys())
        selected_keys = random.sample(keys[:select_len], select_len)

        selected_dict = {key: original_dict[key] for key in selected_keys}
        un_selected_dict = {key: un_original_dict[key] for key in selected_keys}

        sorted_dict = dict(sorted(selected_dict.items(), key=lambda x: float(x[0])))
        un_sorted_dict = dict(sorted(un_selected_dict.items(), key=lambda x: float(x[0])))
        for k, v in sorted_dict.items():
            sorted_dict[k] = v.replace("\n", " ")

        with open(new_file_path, "w+") as fp:
            json.dump(sorted_dict, fp, indent=4)

        with open(un_new_file_path, "w+") as fp:
            json.dump(un_sorted_dict, fp, indent=4)

    file_path_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name,
                                    "raw_js_seq_all.json")
    file_path_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name,
                                     "raw_cmd_seq_all.json")

    file_path_random_picked_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name,
                                                  "raw_js_seq.json")
    file_path_random_picked_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name,
                                                   "raw_cmd_seq.json")

    un_file_path_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name,
                                    "raw_js_seq_all.json")
    un_file_path_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name,
                                     "raw_cmd_seq_all.json")

    un_file_path_random_picked_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name,
                                                  "raw_js_seq.json")
    un_file_path_random_picked_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name,
                                                   "raw_cmd_seq.json")

    select_random_elements_from_json_file_ablation_exm(file_path_js_seq, un_file_path_js_seq, benign_limit,
                                                       file_path_random_picked_js_seq, un_file_path_random_picked_js_seq)
    select_random_elements_from_json_file_ablation_exm(file_path_cmd_seq, un_file_path_cmd_seq, benign_limit,
                                                       file_path_random_picked_cmd_seq, un_file_path_random_picked_cmd_seq)


def get_raw_samples_main_ablation_exm(target_dir, output_file_name_cmd_seq, output_file_name_js_seq, result_dir,
                                      un_output_file_name_cmd_seq, un_output_file_name_js_seq, un_result_dir, mode,
                                      we_train_batch_name):
    pkg_list = check_pkg_metadata(mode, target_dir, result_dir)
    import shutil
    shutil.copy(os.path.join(result_dir, "checked_pkg_list.json"), un_result_dir)
    js_wv_model, cmd_wv_model = load_model(we_train_batch_name)

    target_script_dict = {}
    pkg_name_2_file_path_dict = {}
    pkg_id_2_pkg_name_dict = {}
    count = 0

    for file in pkg_list:
        file_dir = os.path.join(target_dir, file, "package/package.json")
        try:
            with open(file_dir, "r") as f:
                json_data = json.load(f)
                version = json_data["version"]
                pkg_name = json_data["name"] + "-" + version
                if "scripts" in json_data:
                    scripts = json_data["scripts"]
                    pkg_name_2_file_path_dict[pkg_name] = os.path.join(target_dir, file)
                    for name, script in json_data["scripts"].items():
                        # if name == "postinstall" or name == "preinstall":
                        count += 1
                        target_script_dict[count] = scripts[name]
                        pkg_id_2_pkg_name_dict[count] = pkg_name
        except:
            print("fail to open: " + file_dir + "\n")

    my_parser = TrainParser(True, {}, {}, pkg_name_2_file_path_dict, "test", pkg_id_2_pkg_name_dict)
    un_my_parser = TrainParser(False, {}, {}, pkg_name_2_file_path_dict, "test", pkg_id_2_pkg_name_dict)
    my_parser.js_wv_model = js_wv_model
    my_parser.cmd_wv_model = cmd_wv_model
    for key, value in target_script_dict.items():
        print("success: ", pkg_id_2_pkg_name_dict[key])
        root_node = my_parser.parse(value, 'bash')
        my_parser.traverse_bash_ast(root_node, key, we_train_batch_name)

        print("success: ", pkg_id_2_pkg_name_dict[key])
        root_node = un_my_parser.parse(value, 'bash')
        un_my_parser.traverse_bash_ast(root_node, key)

    js_seq_dict = my_parser.js_seq_dict
    cmd_seq_dict = my_parser.cmd_seq_dict

    with open(output_file_name_cmd_seq, 'w+') as json_file:
        json_file.write(json.dumps(cmd_seq_dict, indent=4))
    with open(output_file_name_js_seq, 'w+') as json_file:
        json_file.write(json.dumps(js_seq_dict, indent=4))

    file_name_id2pkg = os.path.join(result_dir, "id2pkg.json")
    with open(file_name_id2pkg, "w+") as f:
        json_str = json.dumps(pkg_id_2_pkg_name_dict, indent=4)
        f.write(json_str)

    un_js_seq_dict = un_my_parser.js_seq_dict
    un_cmd_seq_dict = un_my_parser.cmd_seq_dict

    with open(un_output_file_name_cmd_seq, 'w+') as json_file:
        json_file.write(json.dumps(un_cmd_seq_dict, indent=4))
    with open(un_output_file_name_js_seq, 'w+') as json_file:
        json_file.write(json.dumps(un_js_seq_dict, indent=4))

    file_name_id2pkg = os.path.join(un_result_dir, "id2pkg.json")
    with open(file_name_id2pkg, "w+") as f:
        json_str = json.dumps(pkg_id_2_pkg_name_dict, indent=4)
        f.write(json_str)


def benign_ablation_exm_samples_parser_main(benign_batch_name, un_benign_batch_name, target_dir, benign_limit,
                                            we_train_batch_name):
    if not os.path.exists(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name))):
        os.mkdir(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name)))
    if not os.path.exists(
            os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name))):
        os.mkdir(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name)))

    output_dir = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name)
    file_path_js_seq = os.path.join(output_dir, "raw_js_seq_all.json")
    file_path_cmd_seq = os.path.join(output_dir, "raw_cmd_seq_all.json")

    un_output_dir = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", un_benign_batch_name)
    un_file_path_js_seq = os.path.join(un_output_dir, "raw_js_seq_all.json")
    un_file_path_cmd_seq = os.path.join(un_output_dir, "raw_cmd_seq_all.json")

    get_raw_samples_main_ablation_exm(target_dir, file_path_cmd_seq, file_path_js_seq, output_dir, un_file_path_cmd_seq,
                                      un_file_path_js_seq, un_output_dir, "benign", we_train_batch_name)

    get_random_from_json_ablation_exm(benign_batch_name, un_benign_batch_name, benign_limit)


if __name__ == "__main__":
    model_batch_name = "20240219-1"
    benign_batch_name = "20240224-1"


