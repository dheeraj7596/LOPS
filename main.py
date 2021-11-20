from train import *
import pickle
import json
from sklearn.metrics import classification_report
import torch
from util import *
import copy
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(data_path, dataset, clf, device, filter_flag):
    thresh = 0.6
    num_its = 5

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    with open(data_path + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    if dataset == "books":
        phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))
        label_term_dict = modify_phrases(label_term_dict, phrase_id_map)

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    if dataset != "books":
        df_copy = copy.deepcopy(df)
        df_copy = preprocess(df_copy)
    else:
        df_copy = pickle.load(open(data_path + "df_preprocessed.pkl", "rb"))

    tokenizer = fit_get_tokenizer(df_copy.text, max_words=150000)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    print("Generating pseudo labels..", flush=True)
    X_train_inds, y_train, y_true = generate_pseudo_labels(df_copy, labels, label_term_dict, tokenizer)
    X_test_inds = list(set(range(len(df))) - set(X_train_inds))

    X_train = list(df.iloc[X_train_inds]["text"])
    y_train = [label_to_index[l] for l in y_train]
    y_true = [label_to_index[l] for l in y_true]

    X_test = list(df.iloc[X_test_inds]["text"])
    y_test = list(df.iloc[X_test_inds]["label"])
    y_test = [label_to_index[l] for l in y_test]

    for it in range(num_its):
        temp_label_to_index = {}
        temp_index_to_label = {}

        print("Iteration:", it, flush=True)
        non_train_data = []
        non_train_labels = []
        true_non_train_labels = []

        if filter_flag:
            for i, y in enumerate(sorted(list(set(y_train)))):
                temp_label_to_index[y] = i
                temp_index_to_label[i] = y
            y_train = [temp_label_to_index[y] for y in y_train]

            print("LOPS started..", flush=True)
            X_train, y_train, y_true, non_train_data, non_train_labels, true_non_train_labels = LOPS(X_train,
                                                                                                     y_train,
                                                                                                     y_true,
                                                                                                     clf,
                                                                                                     device)
            y_train = [temp_index_to_label[y] for y in y_train]
            non_train_labels = [temp_index_to_label[y] for y in non_train_labels]
            print("LOPS completed..", flush=True)

        for i in range(len(non_train_data)):
            X_test.append(non_train_data[i])
            y_test.append(true_non_train_labels[i])

        for i, y in enumerate(sorted(list(set(y_train)))):
            temp_label_to_index[y] = i
            temp_index_to_label[i] = y
        y_train = [temp_label_to_index[y] for y in y_train]

        print("Training model..", flush=True)
        model = train_cls(X_train, y_train, clf, device)

        print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************", flush=True)
        predictions = test(model, X_all, y_all_inds, clf, device)
        pred_inds = get_labelinds_from_probs(predictions)
        pred_labels = []
        for p in pred_inds:
            pred_labels.append(index_to_label[temp_index_to_label[p]])
        print(classification_report(y_all, pred_labels), flush=True)
        print("*" * 80, flush=True)

        print("Bootstrapping..", flush=True)

        predictions = test(model, X_test, y_test, clf, device)
        for i, p in enumerate(predictions):
            if i == 0:
                pred = p
            else:
                pred = np.concatenate((pred, p))

        pred_labels = []
        removed_inds = []
        for i, p in enumerate(pred):
            sample = X_test[i]
            true_lbl = y_test[i]
            max_prob = p.max(axis=-1)
            lbl = temp_index_to_label[p.argmax(axis=-1)]
            pred_labels.append(index_to_label[lbl])
            if max_prob >= thresh:
                X_train.append(sample)
                y_train.append(lbl)
                y_true.append(true_lbl)
                removed_inds.append(i)

        removed_inds.sort(reverse=True)
        for i in removed_inds:
            del X_test[i]
            del y_test[i]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/nyt-fine/')
    parser.add_argument('--dataset', type=str, default='nyt-fine')
    parser.add_argument('--clf', type=str, default='bert')
    parser.add_argument('--gpu_id', type=str, default="cpu")
    parser.add_argument('--lops', type=int, default=0)
    args = parser.parse_args()
    if args.gpu_id != "cpu":
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device("cpu")

    main(
        data_path=args.data_path,
        dataset=args.dataset,
        clf=args.clf,
        device=device,
        filter_flag=args.lops
    )
