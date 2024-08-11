import random
from datasets import load_dataset, Dataset
from utils import set_seed
from collections import defaultdict, Counter


def load_hf_data(dataset_name):
    """
    Load HF dataset by mapping internal name to various HF users & datasets.
    """
    dataset_name = dataset_name.replace("v2", "")
    # SuperGLUE
    if dataset_name in ["wic", "boolq"]:
        dataset = load_dataset("super_glue", dataset_name)
    # GLUE
    elif dataset_name in ["cola", "mrpc", "rte"]:
        dataset = load_dataset("glue", dataset_name)
    # other
    elif dataset_name == "reuters":
        dataset = load_dataset("reuters21578", "ModApte")
    elif dataset_name == "emotion":
        dataset = load_dataset("dair-ai/emotion")
    elif dataset_name == "implicithate":
        dataset = load_dataset("SALT-NLP/ImplicitHate")
    elif dataset_name == "sst5":
        dataset = load_dataset("SetFit/sst5")
    elif dataset_name == "sst2":
        dataset = load_dataset("yerevann/sst2")
    elif dataset_name == "stormfront":
        dataset = load_dataset("hate_speech18")
    elif dataset_name == "trec":
        dataset = load_dataset("trec")
    return dataset


def rename_dataset_columns(dataset):
    """
    Ensure all datasets have the sentence(1 + sentence2) --> labels format
    """
    fs = dataset["train"].features
    if "label" in fs:
        dataset = dataset.rename_column("label", "labels")
    if "coarse_label" in fs:
        dataset = dataset.rename_column("coarse_label", "labels")
    if "topics" in fs:
        dataset = dataset.rename_column("topics", "labels")
    if "premise" in fs:
        dataset = dataset.rename_column("premise", "sentence1")
    if "hypothesis" in fs:
        dataset = dataset.rename_column("hypothesis", "sentence2")
    if "text" in fs:
        dataset = dataset.rename_column("text", "sentence")
    if "question" in fs:
        dataset = dataset.rename_column("question", "sentence1")
    if "passage" in fs:
        dataset = dataset.rename_column("passage", "sentence2")
    if "answer" in fs:
        dataset = dataset.rename_column("answer", "labels")
    if "implicit_class" in fs:
        dataset = dataset.rename_column("implicit_class", "labels")
    if "post" in fs:
        dataset = dataset.rename_column("post", "sentence")
    return dataset


def modify_reuters(train, test):
    """
    Only keep top 8 REUTERS classes and examples that have 1 label only.
    """
    classes, _ = zip(
        *Counter([x for t in train["labels"] for x in t if len(t) == 1]).most_common(8)
    )
    remove_indices = []
    for subset in [train, test]:
        for index, label in enumerate(subset["labels"]):
            if (
                len(label) != 1
                or label[0] not in classes
                or subset["sentence"][index] in subset["sentence"][:index]
            ):
                remove_indices.append(index)
            else:
                subset["labels"][index] = label[0]
        for key in subset:
            subset[key] = [
                x for i, x in enumerate(subset[key]) if i not in remove_indices
            ]
    return train, test


def binarize(train, test):
    """
    Only keep 2 most frequent classes.
    """
    labels, _ = zip(*Counter(train["labels"]).most_common(2))
    idxs = [i for i, l in enumerate(train["labels"]) if l in labels]
    for key in train:
        train[key] = [train[key][i] for i in idxs]
    idxs = [i for i, l in enumerate(test["labels"]) if l in labels]
    for key in test:
        test[key] = [test[key][i] for i in idxs]
    train, test = map_labels(train, test)
    return train, test


def map_labels(train, test):
    """
    Change labels that are strings to number classes.
    """
    classes = sorted(list(set(train["labels"])))
    mapping = {v: k for k, v in enumerate(classes)}
    train["labels"] = [mapping[k] for k in train["labels"]]
    test["labels"] = [mapping[k] for k in test["labels"]]
    return train, test


def preprocess_dataset(dataset_name, seed=1, normal=False, ratio=0.15):
    """
    Partially mislabel dataset and create train/val/test split.
    Args:
        - dataset_name (str)
        - seed (int): data seed for different shufflings and mislabellings
        - normal (boolean): if True, no mislabelling happens
        - ratio (float): which portion of the data to mislabel
    Returns:
        - Dataset dict with keys train, validation and test
    """
    set_seed(seed)
    dataset = rename_dataset_columns(load_hf_data(dataset_name))

    train = dataset["train"].to_dict()
    if dataset_name.replace("v2", "") not in ["implicithate", "stormfront"]:
        test = dataset["validation" if "validation" in dataset else "test"].to_dict()
    else:
        # For stormfront remove 2 minusicule classes, should be binary: hate/no
        idxs = [
            i
            for i in list(range(len(train["sentence"])))
            if not (
                dataset_name.replace("v2", "") == "stormfront"
                and train["labels"][i] in [2, 3]
            )
        ]
        random.shuffle(idxs)
        idxs1 = idxs[: int(len(idxs) * 0.8)]
        idxs2 = idxs[int(len(idxs) * 0.8):]
        test = {key: [train[key][i] for i in idxs2] for key in train}
        train = {key: [train[key][i] for i in idxs1] for key in train}

    if dataset_name.replace("v2", "") == "reuters":
        train, test = modify_reuters(train, test)

    if dataset_name.replace("v2", "") in ["reuters", "emotion", "sst5", "implicithate"]:
        train, test = map_labels(train, test)

    if "v2" in dataset_name:
        train, test = binarize(train, test)

    if "idx" not in train or "mrpc" in dataset_name:
        train["idx"] = list(range(len(train["labels"])))
        test["idx"] = list(range(len(test["labels"])))
    elif "idx" in train:
        assert train["idx"] == list(range(len(train["labels"])))

    train["dataset"] = [dataset_name] * len(train["labels"])
    test["dataset"] = [dataset_name] * len(test["labels"])

    # Set aside examples per class to mislabel according to ratio
    classes = sorted(list(set(train["labels"])))
    idxs = defaultdict(lambda: set())
    idx_per_class = {
        c: [i for i, j in enumerate(train["idx"]) if train["labels"][i] == c]
        for c in classes
    }
    for c in idx_per_class:
        random.shuffle(idx_per_class[c])
        idx_per_class[c] = idx_per_class[c][: int(
            len(idx_per_class[c]) * ratio)]

    for i, j in enumerate(train["idx"]):
        if not normal and j in idx_per_class[train["labels"][i]]:
            rand_label = random.choice(
                [c for c in classes if c != train["labels"][i]])
            train["labels"][i] = rand_label
            idxs["noisy"].add(i)
        else:
            idxs["clean"].add(i)

    dataset["train"] = Dataset.from_dict(train)
    dataset["test"] = Dataset.from_dict(test)
    split = dataset["test"].train_test_split(test_size=0.5, shuffle=True)
    dataset["validation"] = split["train"]
    dataset["test"] = split["test"]
    return dataset, dict(idxs)
