import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import torch
import pickle
import sklearn.metrics
from collections import defaultdict
from utils import get_logger, get_args, get_tokenizer, get_model
from data import preprocess_dataset
from utils import collate_tokenize, set_seed


class FeedForwardClassifier(nn.Module):
    def __init__(self, num_labels, predict_noise, dim=768):
        super(FeedForwardClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.in_layer = nn.Linear(dim, 128)
        self.relu = nn.GELU()
        self.hidden_layer = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, num_labels)
        self.predict_noise = predict_noise

    def forward(self, x):
        x = self.relu(self.in_layer(self.dropout(x)))
        x = self.out_layer(self.relu(self.hidden_layer(x)))
        return self.softmax(x)

    def train_(self, X, y, Xv, yv):
        """
        Train for 100 epochs, select model based on validation data.
        Args:
            - X: list of vectors, training data input
            - y: list of ints, training data labels
            - Xv: list of vectors, validation data input
            - yv: list of ints, validation data labels
        """
        # When predicting noise, upweight the noisy labels that only make up
        # 15% of the dataset
        criterion = nn.NLLLoss(
            weight=torch.FloatTensor([0.33, 0.67]).to(device)
            if self.predict_noise else None)
        optimizer = optim.AdamW(
            self.parameters(), lr=0.0002, weight_decay=1e-5)

        best_f1 = 0
        best_epoch = 0
        for epoch in range(100):
            self.train()
            indices = list(range(0, len(X), 32))
            random.shuffle(indices)
            for i in indices:
                inputs, labels = X[i:i+32], y[i:i+32]
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            f1, _ = self.test(Xv, yv)
            if epoch > 25 and f1 > best_f1:
                best_sd = copy.deepcopy(self.state_dict())
                best_f1 = f1
                best_epoch = epoch
            # if epoch >= best_epoch + 50:
            #     break
        if best_f1 != 0:
            self.load_state_dict(best_sd)
        print(best_epoch)

    def test(self, X, y):
        """
        Test probe by computing f1 on test data.
        Args:
            - X: list of vectors, input data
            - y: list of ints, labels
        Returns:
            - f1 (float)
            - prds (list of predicted labels)
        """
        self.eval()
        prds = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                outputs = self.forward(X[i:i+32])
                predicted = torch.argmax(outputs, dim=-1)
                prds.extend(predicted.cpu().tolist())

        if self.predict_noise:
            f1 = sklearn.metrics.f1_score(
                y_true=y.cpu(), y_pred=prds, pos_label=1, average='binary')
        else:
            f1 = sklearn.metrics.f1_score(
                y_true=y.cpu(), y_pred=prds, average='macro')
        return f1, prds


def get_representations(examples, model, tokenizer):
    """
    Pass examples through the model and get the hidden states.
    Args:
        - examples (list): examples come from HF datasets
        - model: instantiation of a HF model class
        - tokenizer: corresponding to model, instantiation of HF AutoTokenizer
    Returns:
        a dict with indices as keys and vectors as values
    """
    reps = defaultdict(list)
    for i, example in examples:
        inputs = collate_tokenize([example], tokenizer)
        if inputs["input_ids"].shape[-1] < 1:
            continue
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "idxs"}
        reps[i] = tuple(x.detach().cpu() for x in model(**inputs)[-1])
    return reps


def get_data_per_layer(examples, reps, layer, predict_noise):
    """
    Extract hidden states for given layer. Create (vector, numerical label,
    str label) combinations.
    Args:
        - examples (list)
        - reps (dict mapping indices to vectors)
        - layer (int)
        - predict_noise (bool): if true predict clean/noisy, else the class
    Returns:
        list of tuples
    """
    dataset = []
    for i, example in examples:
        if i not in reps:
            continue

        vector = reps[i][layer][0][0 if "bert" in model.base else -1].flatten()
        tag = "clean" if i not in idxs["noisy"] else "noisy"
        if predict_noise:
            label = 1 if "noisy" in tag else 0
        else:
            label = example["labels"]
        dataset.append((vector, label, tag))
    return dataset


def probe(logger, dataset, idxs, model, tokenizer, predict_noise, num_labels):
    """
    Train a probing classifier per layer, for all 12 layers, for 5 seeds.
    Args:
        - logger: logging object used to report f1 scores
        - dataset (HF dataset): dataset with partially perturbed labels
        - idxs (dict): gives datapoints indices for `clean` and `noisy`
        - model (HF model class instantiation)
        - tokenizer (AutoTokenizer instantiation)
        - predict_noise (bool): if true predict clean/noisy, else the class
        - num_labels (int): number of output classes
    Returns:
        dict reporting performance per (seed, layer) combination
    """
    indices = list(range(len(dataset["train"])))
    examples = list(zip(indices, dataset["train"]))
    model.eval()
    reps = get_representations(examples, model, tokenizer)
    accs = defaultdict(list)
    for seed in range(5):
        logger.info(f"Seed {seed}")
        for layer in range(13):
            dataset = get_data_per_layer(examples, reps, layer, predict_noise)
            set_seed(seed)
            random.shuffle(dataset)
            n = int(len(dataset) * 0.7)
            m = int(len(dataset) * 0.8)
            train, valid, test = dataset[:n], dataset[n:m], dataset[m:]

            # Train on 70%, validate on 10%
            X, y, _ = zip(*train)
            Xv, yv, _ = zip(*valid)
            classifier = FeedForwardClassifier(num_labels, predict_noise)
            classifier.to(device)
            classifier.train_(torch.stack(X).to(device),
                              torch.LongTensor(y).to(device),
                              torch.stack(Xv).to(device),
                              torch.LongTensor(yv).to(device))

            # Evaluate on full train & test set
            train_f1, _ = classifier.test(torch.stack(X).to(
                device), torch.LongTensor(y).to(device))
            X, y, tags = zip(*test)
            test_f1, prds = classifier.test(
                torch.stack(X).to(device), torch.LongTensor(y).to(device))
            logger.info(f"- {layer}: {train_f1:.3f} /"
                        + f" {test_f1:.3f} / {np.mean(y)} / {np.mean(prds)}")

            # Test noisy & clean examples separately
            X, y, tags = zip(*[t for t in test if t[-1] == "noisy"])
            test_f1_noisy, prds = classifier.test(
                torch.stack(X).to(device), torch.LongTensor(y).to(device))
            X, y, tags = zip(*[t for t in test if t[-1] == "clean"])
            test_f1_clean, prds = classifier.test(
                torch.stack(X).to(device), torch.LongTensor(y).to(device))
            accs[seed, layer] = {
                "all": test_f1, "noisy": test_f1_noisy, "clean": test_f1_clean}
    return accs


if __name__ == "__main__":
    logger = get_logger()
    args = get_args(logger)

    # Load the data, model and corresponding tokenizer
    dataset, idxs = preprocess_dataset(
        args.dataset, args.data_seed, normal=False)
    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, args.custom_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train a probe to predict whether or not a certain example is noisy
    all_results = dict()
    num_labels = model.classifier.weight.shape[0]
    logger.info("Predict noise")
    all_results["predict_noise"] = probe(
        logger, dataset, idxs, model, tokenizer,
        predict_noise=True, num_labels=2)

    if "embeddings_" in args.custom_model:
        # Train a probe to predict the labels of the perturbed dataset
        logger.info("Predict label")
        all_results["predict_noisy_label"] = probe(
            logger, dataset, idxs, model, tokenizer,
            predict_noise=False, num_labels=num_labels)

        # Now reload the data, but with all labels "clean"
        dataset, _ = preprocess_dataset(
            args.dataset,  args.data_seed, normal=True)
        logger.info("Predict label")
        all_results["predict_clean_label"] = probe(
            logger, dataset, idxs, model, tokenizer,
            predict_noise=False, num_labels=num_labels)

    folder = args.checkpoint_folder.replace("checkpoints", "results")
    model = args.custom_model.split(f'checkpoints/{args.dataset}/')[-1]
    pickle.dump(
        all_results,
        open(f"{folder}/probing/{args.dataset}/{model}.pickle", 'wb'))
