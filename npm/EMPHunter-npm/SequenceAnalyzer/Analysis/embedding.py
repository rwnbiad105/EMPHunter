from SequenceAnalyzer.Model.Sentence_Embedding.SE_Model import SE_Model
from SequenceAnalyzer.Initialize.prepare_samples import prepare_samples, prepare_basedata_samples
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def embedding_main(batch_name, result_dir, train_batch_name):
    my_se_model = SE_Model()
    prepare_samples(batch_name, result_dir)
    target_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/Result/", batch_name, "labeled_js_seq.txt")
    target_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/Result/", batch_name, "labeled_cmd_seq.txt")
    output_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/Result/", batch_name, "js_seq_vec")
    output_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/Result/", batch_name, "cmd_seq_vec")
    my_se_model.embed_seq(target_dir1, train_batch_name, output_dir1, "js_model.bin")
    my_se_model.embed_seq(target_dir2, train_batch_name, output_dir2, "cmd_model.bin")


def embedding_basedata(basedata_batch_name, train_batch_name, basedata_mode):
    my_se_model = SE_Model()
    labeled_js_seq_fname = "labeled_js_seq.txt"
    labeled_cmd_seq_fname = "labeled_cmd_seq.txt"
    basedata_sample_dir = os.path.join(CURRENT_DIR, "../../DataShare/BaseData", basedata_mode, basedata_batch_name)
    prepare_basedata_samples("raw_js_seq.json", labeled_js_seq_fname, basedata_sample_dir)
    prepare_basedata_samples("raw_cmd_seq.json", labeled_cmd_seq_fname, basedata_sample_dir)
    target_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, labeled_js_seq_fname)
    target_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, labeled_cmd_seq_fname)
    output_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "js_seq_vec")
    output_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "cmd_seq_vec")
    my_se_model.embed_seq(target_dir1, train_batch_name, output_dir1, "js_model.bin")
    my_se_model.embed_seq(target_dir2, train_batch_name, output_dir2, "cmd_model.bin")


def embed_a_batch(basedata_mode, basedata_batch_name, my_se_model, train_batch_name, count):
    target_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul", "labeled_js_seq_" + str(count) + ".txt")
    target_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul", "labeled_cmd_seq_" + str(count) + ".txt")
    output_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul", "js_seq_vec_" + str(count))
    output_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul", "cmd_seq_vec_" + str(count))
    my_se_model.embed_seq(target_dir1, train_batch_name, output_dir1, "js_model.bin", count)
    my_se_model.embed_seq(target_dir2, train_batch_name, output_dir2, "cmd_model.bin", count)


def mult_process_embedding_basedata(basedata_batch_name, train_batch_name, basedata_mode, workers):
    import threading
    my_se_model = SE_Model()
    basedata_sample_dir = os.path.join(CURRENT_DIR, "../../DataShare/BaseData", basedata_mode, basedata_batch_name)
    prepare_basedata_samples("raw_js_seq.json", "labeled_js_seq.txt", basedata_sample_dir)
    prepare_basedata_samples("raw_cmd_seq.json", "labeled_cmd_seq.txt", basedata_sample_dir)

    if not os.path.exists(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name,"ForMul")):
        os.mkdir(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name,"ForMul"))

    with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "labeled_js_seq.txt"), "r") as fp:
        js_lines = fp.readlines()
        js_len = len(js_lines)
        batch_len = js_len // workers
        rest = js_len % workers
    for count in range(1, workers+1):
        if count == workers:
            batch_len = batch_len + rest
        with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul",
                               "labeled_js_seq_" + str(count) + ".txt"), "w+") as fp:
            fp.writelines(js_lines[(count-1)*batch_len:(count-1)*batch_len + batch_len])

    with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "labeled_cmd_seq.txt"), "r") as fp:
        cmd_lines = fp.readlines()
        cmd_len = len(cmd_lines)
        batch_len = cmd_len // workers
        rest = cmd_len % workers
    for count in range(1, workers+1):
        if count == workers:
            cur_batch_len = batch_len + rest
        else:
            cur_batch_len = batch_len
        with open(os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "ForMul",
                               "labeled_cmd_seq_" + str(count) + ".txt"), "w+") as fp:
            fp.writelines(cmd_lines[(count-1)*batch_len:(count-1)*batch_len + cur_batch_len])

    threads = []
    for count in range(1, workers+1):
        t = threading.Thread(target=embed_a_batch, args=(basedata_mode, basedata_batch_name, my_se_model, train_batch_name, count))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    js_vec_list = []
    cmd_vec_list = []
    for count in range(1, workers+1):
        output_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name,
                                   "ForMul", "js_seq_vec_" + str(count) + ".vec")
        output_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name,
                                   "ForMul", "cmd_seq_vec_" + str(count) + ".vec")
        with open(output_dir1, "r") as fp:
            js_lines = fp.readlines()
            js_vec_list += js_lines
        with open(output_dir2, "r") as fp:
            cmd_lines = fp.readlines()
            cmd_vec_list += cmd_lines

    output_dir1 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "js_seq_vec")
    output_dir2 = os.path.join(CURRENT_DIR, "../../DataShare/BaseData/", basedata_mode, basedata_batch_name, "cmd_seq_vec")
    with open(output_dir1 + ".vec", "w+") as fp:
        fp.writelines(js_vec_list)
    with open(output_dir2 + ".vec", "w+") as fp:
        fp.writelines(cmd_vec_list)


if __name__ == "__main__":

    result_dir = os.path.join(CURRENT_DIR, "../../DataShare/Result")
    js_model_name = "20240219-js"
    cmd_model_name = "20240222-cmd"

    batch_name = "20240203-2"
    embedding_main(batch_name, result_dir, js_model_name, cmd_model_name)

    # benign_batch_name = "20240224-1"
    # embedding_benign(benign_batch_name, js_model_name, cmd_model_name)
