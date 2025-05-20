import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class SE_Model:
    def __init__(self, model_name="test", epoch=200, dim=300, output_name="output_simplified.txt"):
        self.epoch = epoch
        self.dim = dim
        self.model_name = model_name
        self.output_name = output_name

    def train(self, train_batch_name, train_sample_file_name, model_name):
        train_sample_dir = os.path.join(CURRENT_DIR, "../../../DataShare/Model/SeqEmbedding", train_batch_name)
        model_config = os.path.join(train_sample_dir, model_name) + "_model -epoch "\
                       + str(self.epoch) + " -dim " + str(self.dim)
        # if os.path.exists(model_dir):
        #     os.mkdir(model_dir)
        cmd = "cd " + PROJECT_DIR + "/SequenceAnalyzer/Model/fastText-PV\n" \
                                    "./fasttext PVDM -input " + os.path.join(CURRENT_DIR, train_sample_dir, train_sample_file_name + "_labeled.txt")\
                                 + " -output " + model_config
        print(cmd)
        os.system(cmd)

    def embed_single_seq(self, target_temp_dir, output_temp_dir, train_batch_name, model_name):
        model_config = PROJECT_DIR + "/DataShare/Model/SeqEmbedding/" + train_batch_name + "/" + model_name
        cmd = "cd " + PROJECT_DIR + "/SequenceAnalyzer/Model/fastText-PV && ./fasttext predictPVDM " + model_config \
              + " " + target_temp_dir + " " + output_temp_dir
        print(cmd)
        os.system(cmd)

    def embed_seq(self, target_dir, train_batch_name, output_dir, model_name, count=None):

        target_parents_dir = "/".join(target_dir.split("/")[:-1])
        output_parents_dir = "/".join(output_dir.split("/")[:-1])
        with open(target_dir, "r") as f:
            target_list = f.readlines()
        vec_list = []

        if count:
            tmp_txt_name = "temp_" + str(count) + ".txt"
            tmp_vec_name = "temp_" + str(count)
        else:
            tmp_txt_name = "temp.txt"
            tmp_vec_name = "temp"
        for i in target_list:
            with open(os.path.join(target_parents_dir, tmp_txt_name), "w+") as f:
                f.write(i)
            self.embed_single_seq(os.path.join(target_parents_dir, tmp_txt_name),
                             os.path.join(output_parents_dir, tmp_vec_name), train_batch_name, model_name)
            with open(os.path.join(output_parents_dir, tmp_vec_name + ".vec"), "r+") as f2:
                vec = f2.read()
                vec_list.append(vec)

        with open(output_dir + ".vec", "w+") as f:
            for vec in vec_list:
                f.write(vec)


if __name__ == "__main__":
    my_se_model = SE_Model()
    result_dir = "../../../DataShare/Result"
    train_sample_dir = "../../../DataShare/Model/SeqEmbedding"
    train_sample_file_name = "train_samples_20240219-1"
    model_name = "20240219"
    # prepare_samples(batch_name, result_dir)
    # my_se_model.prepare_train_samples(train_sample_file_name, train_sample_dir)
    my_se_model.train(train_sample_dir, train_sample_file_name, model_name)