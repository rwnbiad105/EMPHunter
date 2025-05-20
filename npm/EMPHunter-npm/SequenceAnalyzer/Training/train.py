from SequenceAnalyzer.Model.Sentence_Embedding.SE_Model import SE_Model
from SequenceAnalyzer.Initialize.prepare_samples import prepare_train_samples
from SequenceAnalyzer.Model.Word_Embedding.Train_model import train, compute_wem_centroid
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_se(train_batch_name, train_sample_file_name, model_name):
    my_se_model = SE_Model(epoch=200)
    prepare_train_samples(train_batch_name, train_sample_file_name)
    my_se_model.train(train_batch_name, train_sample_file_name, model_name)


def train_we(model_batch_name, wv_model_name, train_sample_file_name, centroid_file_name):
    vector_size = 300
    window = 5
    min_count = 1
    epochs = 200
    sg = 1
    workers = 16
    corpus_file_path = os.path.join(CURRENT_DIR, "../../DataShare/Model/WordEmbedding/", model_batch_name, train_sample_file_name)
    train(vector_size, window, min_count, epochs, sg, workers, model_batch_name, wv_model_name, corpus_file_path)
    compute_wem_centroid(model_batch_name, wv_model_name, centroid_file_name)


if __name__ == "__main__":
    train_batch_name = "20240219-1"
    train(train_batch_name, "cmd_train_samples", "cmd")
    train(train_batch_name, "js_train_samples", "js")

