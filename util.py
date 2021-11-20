import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import string
import random
from transformers import BertTokenizerFast, GPT2TokenizerFast, RobertaTokenizerFast, XLNetTokenizerFast, \
    BertForSequenceClassification, GPT2ForSequenceClassification, RobertaForSequenceClassification, \
    XLNetForSequenceClassification


def generate_name(id):
    return "fnust" + str(id)


def modify_phrases(label_term_dict, phrase_id_map, random_k=0):
    for l in label_term_dict:
        temp_list = []
        for term in label_term_dict[l]:
            try:
                temp_list.append(generate_name(phrase_id_map[term]))
            except:
                temp_list.append(term)
        if random_k:
            random.shuffle(temp_list)
            label_term_dict[l] = temp_list[:random_k]
        else:
            label_term_dict[l] = temp_list
    return label_term_dict


def preprocess(df):
    print("Preprocessing data..", flush=True)
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)), flush=True)
        line = row["text"]
        words = line.strip().split()
        new_words = []
        for word in words:
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if len(word_clean) == 0 or word_clean in stop_words:
                continue
            new_words.append(word_clean)
        df["text"][index] = " ".join(new_words)
    return df


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        keys = sorted(count_dict.keys())
        for l in keys:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(index)
            y_true.append(label)
    return X, y, y_true


def get_labelinds_from_probs(predictions):
    for i, p in enumerate(predictions):
        if i == 0:
            pred = p
        else:
            pred = np.concatenate((pred, p))
    pred_inds = []
    for p in pred:
        pred_inds.append(p.argmax(axis=-1))
    return pred_inds


def compute_train_non_train_inds(pred_inds, true_inds, inds_map, lbl):
    train_inds = []
    non_train_inds = []
    for ind in inds_map[lbl]:
        if pred_inds[ind] == true_inds[ind]:
            train_inds.append(ind)
        else:
            non_train_inds.append(ind)
    return train_inds, non_train_inds


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer


def create_label_index_maps(labels):
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return label_to_index, index_to_label


def get_model_config(clf, num_labels=2):
    if clf == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', do_lower_case=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        model.config.pad_token_id = model.config.eos_token_id
        num_epochs = 4
        batch_size = 4
    elif clf == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        num_epochs = 4
        batch_size = 32
    elif clf == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        num_epochs = 3
        batch_size = 32
    elif clf == "xlnet":
        tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)
        model = XLNetForSequenceClassification.from_pretrained(
            "xlnet-base-cased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        num_epochs = 4
        batch_size = 32
    else:
        tokenizer = None
        model = None
        num_epochs = None
    return tokenizer, model, num_epochs, batch_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
