import torch
import random
import numpy as np
import argparse
import logging
from models.bert import BertForSequenceClassification
from models.opt import OPTForSequenceClassification
from models.gptneo import GPTNeoForSequenceClassification
from models.gptneox import GPTNeoXForSequenceClassification
from sklearn.metrics import accuracy_score as accuracy, f1_score as f1
from transformers import AutoTokenizer, AutoConfig


def collate_tokenize(data, tok):
    """
    Encode a batch of examples using the given tokenizer.
    Args:
        - data: batches from datasets.DataLoader object
        - tok: transformers.AutoTokenizer
    """
    text = []
    septoken = tok.sep_token

    # Dataset specific formatting
    for example in data:
        if example["dataset"] == "wic":
            text.append(f"{example['word']} {septoken} {example['sentence1']}"
                        + f" {septoken} {example['sentence2']}")
        elif "sentence" in example:
            text.append(example['sentence'])
        elif "sentence1" in example:
            text.append(
                f"{example['sentence1']} {septoken} {example['sentence2']}")
        else:
            raise NotImplementedError(
                f"Missing {example['dataset']} tokenisation")

    # Encode text with tokens, add labels and datapoint indices
    tokenized = tok(text, padding='longest',
                    truncation=True, return_tensors='pt',
                    add_special_tokens=True, max_length=512)
    for k in tokenized:
        tokenized[k] = torch.LongTensor(tokenized[k].long())
    tokenized["labels"] = torch.LongTensor(
        [x['labels'] for x in data])
    tokenized["idxs"] = torch.LongTensor([x["idx"] for x in data])
    return tokenized


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name.replace("_", "/"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        if "opt" in model_name.lower():
            tokenizer.sep_token = tokenizer.eos_token
        else:
            # Use a different sep token for GPT-Neo and Pythia:
            # they failed at WiC when using the EOS as SEP token
            tokenizer.sep_token = "|"
    tokenizer.max_length = 512
    return tokenizer


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate(eval_dataloader, model, device, auxiliary=False):
    """
    Evaluate the model while collecting output logits.
    When using output logits: make sure to give unshuffled dataloader.
    Args:
        - eval_dataloader: datasets.DataLoader object
        - model: HF model
        - device: cpu/gpu
        - auxiliary (bool): whether the dataloader is for the auxiliary task
    Returns:
        - dictionary with f1 and accuracy metrics as keys
        - list of logits for all examples in the (unshuffled!) dataset
    """
    all_prd, all_labels, all_idxs = [], [], []

    model.eval()
    all_logits = []
    for batch in eval_dataloader:
        all_idxs.extend(batch["idxs"].tolist())
        batch = {k: v.to(device) for k, v in batch.items() if k != "idxs"}
        with torch.no_grad():
            outputs = model(**batch, auxiliary=auxiliary)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_logits.extend(outputs.logits.cpu().numpy().tolist())
        all_prd.extend(predictions.tolist())
        all_labels.extend(batch["labels"].tolist())

    perf = {"accuracy": accuracy(y_true=all_labels, y_pred=all_prd),
            "f1": f1(y_true=all_labels, y_pred=all_prd, average="macro")}
    return perf, all_logits


def get_logger():
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_model_family(name):
    """
    Return the right HF model class based on input name.
    Args:
        - name (str)
    Returns:
        - model class, one of BertForSequenceClassification,
        OPTForSequenceClassification, GPTNeoXForSequenceClassification,
        GPTNeoForSequenceClassification
    """
    if "bert" in name.lower():
        return BertForSequenceClassification
    elif "opt" in name.lower():
        return OPTForSequenceClassification
    elif "pythia" in name.lower():
        return GPTNeoXForSequenceClassification
    elif "neo" in name.lower():
        return GPTNeoForSequenceClassification
    raise NotImplementedError(f"Don't know {name}...")


def get_model(model_name, custom_model):
    config = AutoConfig.from_pretrained(
        model_name.replace("_", "/"), output_hidden_states=True)
    model_state_dict = torch.load(
        custom_model, map_location=torch.device('cpu'))[0]
    config.num_labels = model_state_dict["classifier.weight"].shape[0]
    config.num_labels2 = model_state_dict["classifier2.weight"].shape[0]
    config.pad_token_id = config.eos_token_id
    config.learn_subspaces = False
    config.cog_training = False
    model = get_model_family(model_name.replace("_", "/")).from_pretrained(
        model_name.replace("_", "/"), config=config)
    model.load_state_dict(model_state_dict, strict=False)
    model.base = model_name
    return model


def get_args(logger):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--aux_dataset", type=str, default="")
    parser.add_argument("--freeze", type=str, nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--custom_model", type=str)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_folder", type=str,
                        default="/home/s2112866/reproductions/memorisation_localisation/checkpoints")
    parser.add_argument("--training_type", type=str,
                        choices=["mmap", "regular", "freeze_train"],
                        default="regular")
    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        choices=["facebook_opt-125m",
                                 "facebook_opt-1.3b",
                                 "bert-base-cased",
                                 "EleutherAI_pythia-160m-deduped",
                                 "EleutherAI_gpt-neo-125m"])
    parser.add_argument("--normal", action="store_true")
    args = parser.parse_args()
    logger.info("Arguments passed to argparse:")
    for k, v in vars(args).items():
        logger.info(f"     - {k}: {v}")
    return args
