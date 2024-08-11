import pickle
import numpy as np
import torch
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoConfig
from data import preprocess_dataset
from utils import (
    collate_tokenize,
    get_args,
    get_logger,
    get_model_family,
    get_tokenizer,
)


def mix(model_mem, model_nonmem, nonmem_layers, model_name):
    """
    Combine the weights of model_mem with model_nonmem.
    Args:
        - model_mem: the base model
        - model_nonmem: the model from which to take certain layers
        - nonmem_layers (list): layers to take from the nonmem model
        - model_name (str): HF model class
    Returns:
        state dict
    """
    new_model = dict()
    for k in model_mem:
        if "h" == k.split(".")[1] or "layers" in k or "layer." in k:
            if "opt" in model_name.lower() or "bert" in model_name.lower():
                layer = int(k.split(".")[3])
            else:
                layer = int(k.split(".")[2])

            if layer in nonmem_layers:
                new_model[k] = model_nonmem[k]
            else:
                new_model[k] = model_mem[k]
        else:
            new_model[k] = model_mem[k]
    return new_model


def validate(model, dataloader, device):
    """
    Compute accuracy on given data.
    Args:
        - model (instance of HF class)
        - dataloader: torch.utils.data.DataLoader instance
        - device: torch.device( cpu | cuda ), device that model is on
    Returns:
        accuracy (float)
    """
    with torch.no_grad():
        prd, lbl = [], []
        for b in dataloader:
            b = {k: v.to(device) for k, v in b.items() if k != "idxs"}
            prd.extend(model(**b).logits.argmax(dim=-1).flatten().tolist())
            lbl.extend(b["labels"].tolist())
    return np.sum([x_ == y_ for x_, y_ in zip(prd, lbl)]) / len(prd)


def main(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nonmem = torch.load(
        args.custom_model.replace("_epoch", "_normal_epoch"), map_location=device
    )[0]
    model_mem = torch.load(args.custom_model, map_location=device)[0]

    # Initialise model
    tokenizer = get_tokenizer(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name.replace("_", "/"))
    config.learn_subspaces = False
    config.cog_training = False
    config.num_labels = model_mem["classifier.weight"].shape[0]
    config.num_labels2 = model_mem["classifier2.weight"].shape[0]
    config.pad_token_id = config.eos_token_id
    for k in list(model_nonmem.keys()):
        if "classifier2" in k:
            del model_nonmem[k]
            del model_mem[k]
    model = get_model_family(args.model_name).from_pretrained(
        args.model_name.replace("_", "/"), config=config
    )

    # Dataloader for noisy & clean data
    dataset, idxs = preprocess_dataset(args.dataset, 1)
    logger.info(str(list(idxs["noisy"])[:10]))
    cfn = partial(collate_tokenize, tok=tokenizer)
    noisy_dataloader = DataLoader(
        [dataset["train"][j] for j in list(idxs["noisy"])], batch_size=8, collate_fn=cfn
    )
    clean_dataloader = DataLoader(
        [dataset["train"][j] for j in list(idxs["clean"])], batch_size=8, collate_fn=cfn
    )

    # Get consecutive layers of varying window sizes, our retrain configs
    maxi = 12 if "1.3b" not in args.model_name else 24
    retrain_configs = sorted(
        list(
            set(
                [
                    tuple(range(i, min(i + nlayers, maxi)))
                    for nlayers in range(1, maxi + 1)
                    for i in range(0, maxi)
                ]
            )
        )
    )

    # Compute accuracies per train config for noisy & clean data
    results = defaultdict(list)
    for layer_combination in retrain_configs:
        logger.info(layer_combination)
        model.load_state_dict(
            mix(model_mem, model_nonmem, layer_combination, args.model_name),
            strict=False,
        )
        model.to(device)
        model.eval()
        acc_noisy = validate(model, noisy_dataloader, device)
        acc_clean = validate(model, clean_dataloader, device)
        logger.info(
            f"{args.dataset}, {layer_combination}, {acc_noisy}, {acc_clean}")
        results[layer_combination] = (acc_noisy, acc_clean)
    return results


if __name__ == "__main__":
    logger = get_logger()
    args = get_args(logger)

    results = main(args, logger)
    folder = args.checkpoint_folder.replace("checkpoints", "results")
    model = args.custom_model.split(f"checkpoints/{args.dataset}/")[-1]
    pickle.dump(
        results, open(
            f"{folder}/layer_swapping/{args.dataset}/{model}.pickle", "wb")
    )
