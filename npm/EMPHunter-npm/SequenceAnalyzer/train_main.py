from SequenceAnalyzer.Training.train import train_se, train_we


def train_all(train_batch_name):
    train_se(train_batch_name, "cmd_train_samples", "cmd")
    train_se(train_batch_name, "js_train_samples", "js")


def train_wv_model(we_train_batch_name):
    train_we(we_train_batch_name, 'js_we.model', "js_train_samples.json", "js_centroid_vec.pkl")
    train_we(we_train_batch_name, 'cmd_we.model', "cmd_train_samples.json", "cmd_centroid_vec.pkl")


if __name__ == "__main__":
    train_batch_name = "20240219-1"
    train_all(train_batch_name)