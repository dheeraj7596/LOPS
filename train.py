from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import time
import random
import datetime
from util import *
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def tokenize(tokenizer, sentences, labels):
    temp = tokenizer(
        sentences,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    input_ids = temp["input_ids"]
    attention_masks = temp["attention_mask"]
    labels = torch.tensor(labels)
    # Print sentence 0, now as a list of IDs.
    # print('Original: ', sentences[0])
    # print('Token IDs:', input_ids[0])
    return input_ids, attention_masks, labels


def create_data_loaders(dataset, batch_size):
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
        pin_memory=True,
        num_workers=16
    )
    return train_dataloader, validation_dataloader


def train(model, train_dataloader, validation_dataloader, device, epochs):
    model = model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy), flush=True)

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    return model


def evaluate(model, prediction_dataloader, device):
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels


def test(model, X_test, y_test, clf, device):
    tokenizer, _, _, batch_size = get_model_config(clf)
    input_ids, attention_masks, labels = tokenize(tokenizer, X_test, y_test)
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )
    predictions, true_labels = evaluate(model, prediction_dataloader, device)
    return predictions


def train_cls(X, y, clf, device):
    num_labels = len(set(y))
    tokenizer, model, num_epochs, batch_size = get_model_config(clf, num_labels)
    input_ids, attention_masks, labels = tokenize(tokenizer, X, y)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size)

    # Tell pytorch to run this model on the GPU.

    model = train(model,
                  train_dataloader,
                  validation_dataloader,
                  device,
                  num_epochs
                  )
    return model


def LOPS(X, y_pseudo, y_true, clf, device):
    percent_thresh = 0.5
    num_labels = len(set(y_pseudo))
    tokenizer, model, epochs, batch_size = get_model_config(clf, num_labels)

    input_ids, attention_masks, labels = tokenize(tokenizer, X, y_pseudo)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )

    stop_flag = False
    inds_map = {}
    for i, j in enumerate(y_pseudo):
        try:
            inds_map[j].append(i)
        except:
            inds_map[j] = [i]

    thresh_map = dict(Counter(y_pseudo))
    print("Counts of pseudo-labels ", thresh_map, flush=True)
    for i in thresh_map:
        thresh_map[i] = int(thresh_map[i] * percent_thresh)

    print("Threshold map ", thresh_map, flush=True)

    filter_flag_map = {}
    train_inds_map = {}
    non_train_inds_map = {}
    for i in thresh_map:
        filter_flag_map[i] = False
        train_inds_map[i] = []
        non_train_inds_map[i] = []

    model = model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    training_stats = []
    total_t0 = time.time()

    epoch_i = 0
    while not stop_flag and epoch_i < epochs:
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        t0 = time.time()
        total_train_loss = 0

        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        prediction_sampler = SequentialSampler(dataset)
        prediction_dataloader = DataLoader(dataset,
                                           sampler=prediction_sampler,
                                           batch_size=batch_size,
                                           num_workers=16,
                                           pin_memory=True
                                           )

        ep_preds, ep_true_labels = evaluate(model, prediction_dataloader, device)
        ep_pred_inds = get_labelinds_from_probs(ep_preds)

        count = 0
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                train_inds, non_train_inds = compute_train_non_train_inds(ep_pred_inds, y_pseudo, inds_map, i)
                train_inds_map[i] = train_inds
                non_train_inds_map[i] = non_train_inds
                if len(train_inds) >= thresh_map[i]:
                    filter_flag_map[i] = True
                    count += 1
            else:
                count += 1

        print("Number of labels reached 50 percent threshold", count)
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                print("For label ", i, " Number expected ", thresh_map[i], " Found ", len(train_inds_map[i]))

        temp_flg = True
        for i in filter_flag_map:
            temp_flg = temp_flg and filter_flag_map[i]
        stop_flag = temp_flg
        epoch_i += 1

    if not stop_flag:
        print("MAX EPOCHS REACHED!!!!!!", flush=True)
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                print("Resetting train, non-train inds for label ", i)
                train_inds_map[i] = inds_map[i]
                non_train_inds_map[i] = []

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    for lbl in train_inds_map:
        for loop_ind in train_inds_map[lbl]:
            train_data.append(X[loop_ind])
            train_labels.append(y_pseudo[loop_ind])
            true_train_labels.append(y_true[loop_ind])

    for lbl in non_train_inds_map:
        for loop_ind in non_train_inds_map[lbl]:
            non_train_data.append(X[loop_ind])
            non_train_labels.append(y_pseudo[loop_ind])
            true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels
