from gensim.models import FastText
from gensim.models import Word2Vec
from gensim.test.utils import datapath
import os
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(vector_size=300, window=5, min_count=1, epochs=50, sg=1, workers=12, model_batch_name="test",
          wv_model_name="test", corpus_file_path="output_simplified.txt"):
    if not os.path.exists(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding")):
        os.mkdir(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding", model_batch_name)):
        os.mkdir(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding", model_batch_name))
    corpus_file = datapath(corpus_file_path)
    #corpus_file = datapath('/home/osroot/api_sequences.txt')

    model = FastText(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sg=sg, workers=workers)

    model.build_vocab(corpus_file=corpus_file)

    total_words = model.corpus_total_words
    print("\nstart training....")
    model.train(corpus_file=corpus_file,total_words=total_words,epochs=5)
    print("done")
    print("saving the model...")
    model.save(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding", model_batch_name, wv_model_name))
    print("done\n")


def compute_wem_centroid(model_batch_name, train_sample_file_name, centroid_file_name):
    model = FastText.load(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding", model_batch_name,
                                       train_sample_file_name))

    centroid_vec = None
    # print(model.wv.vector_size)
    # print(model.corpus_total_words)
    for key, index in model.wv.key_to_index.items():
        if centroid_vec is None:
            centroid_vec = model.wv[key].copy()
            # print(type(centroid_vec))
        else:
            centroid_vec += model.wv[key].copy()

    total_len = model.corpus_total_words

    with open(os.path.join(CURRENT_DIR, "../../../DataShare/Model/WordEmbedding", model_batch_name, centroid_file_name),
              "wb+") as fp:
        pickle.dump(centroid_vec/total_len, fp)


if __name__ == "__main__":
    vector_size = 300
    window = 5
    min_count = 1
    epochs = 5
    sg = 1
    workers = 12
    model_batch_name = "20240406-1"
    wv_model_name = "300-win5-mc1"
    mode = "ft"
    corpus_file_path = os.path.join(CURRENT_DIR, "../../../DataShare/Model/SeqEmbedding/test-20240331-1/js_train_samples_labeled.txt")

    # train(vector_size, window, min_count, epochs, sg, workers, model_batch_name, mode, corpus_file_path)
    compute_wem_centroid(model_batch_name)