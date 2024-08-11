import torch
import pickle
import random
import os
from transformers import AutoConfig
from torch.utils.data import DataLoader
from functools import partial
from data import preprocess_dataset
from main import train
from utils import (
    set_seed,
    validate,
    collate_tokenize,
    get_logger,
    get_args,
    get_model_family,
    get_tokenizer,
)


def main(args, seed, logger, retrain):
    dataset, idxs = preprocess_dataset(args.dataset, 1)
    tokenizer = get_tokenizer(args.model_name)
    set_seed(seed)

    # Use as many clean examples as there are noisy examples for testing
    # Use the remainder of the clean examples for the retraining
    idxs["clean"] = list(idxs["clean"])
    random.shuffle(idxs["clean"])
    other_test = idxs["clean"][: len(idxs["noisy"])]
    other_train = idxs["clean"][len(idxs["noisy"]):]

    # Create DataLoader objects for train and eval (other and noisy examples)
    cfn = partial(collate_tokenize, tok=tokenizer)
    train_dataloader = DataLoader(
        [dataset["train"][i] for i in other_train],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=cfn,
    )
    eval_other = DataLoader(
        [dataset["train"][i] for i in other_test],
        batch_size=args.batch_size,
        collate_fn=cfn,
    )
    eval_noisy = DataLoader(
        [dataset["train"][i] for i in idxs["noisy"]],
        batch_size=args.batch_size,
        collate_fn=cfn,
    )

    # The layers to freeze are the ones you do not want to retrain
    freeze = [
        l
        for l in [
            "embeddings",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "classifier",
            "pooler",
        ]
        if l not in retrain
    ]

    state_dict = torch.load(
        args.custom_model, map_location=torch.device("cpu"))[0]
    config = AutoConfig.from_pretrained(args.model_name.replace("_", "/"))
    config.learn_subspaces = False
    config.cog_training = False
    config.num_labels = state_dict["classifier.weight"].shape[0]
    config.num_labels2 = state_dict["classifier2.weight"].shape[0]
    config.pad_token_id = config.eos_token_id
    model = get_model_family(args.model_name).from_pretrained(
        args.model_name.replace("_", "/"), config=config
    )
    model.base = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model but exclude items from state dict that were listed as
    # layers to retrain
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(
            [
                f"layer.{l}." in k or f"layers.{l}." in k or f"transformer.h.{l}." in k
                for l in retrain
            ]
        )
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Now use the regular training function for 5 epochs
    train(
        logger,
        model,
        device,
        train_dataloader,
        eval_noisy,
        5,
        args.lr,
        freeze=freeze,
        store=False,
    )

    p1, _ = validate(train_dataloader, model, device)
    p2, _ = validate(eval_noisy, model, device)
    p3, _ = validate(eval_other, model, device)
    logger.info(
        f"Final train acc = {p1['accuracy']:.3f}, f1 = {p1['f1']:.3f}, "
        + f"val acc noisy = {p2['accuracy']:.3f}, f1 = {p2['f1']:.3f}, "
        + f"val acc clean = {p3['accuracy']:.3f}, f1 = {p3['f1']:.3f}"
    )
    return p1, p2, p3


if __name__ == "__main__":
    logger = get_logger()
    args = get_args(logger)

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

    # Construct path for saving the results pickle
    root = args.checkpoint_folder.replace("checkpoints", "results")
    folder = "layer_retraining"
    model = args.custom_model.split(f"checkpoints/{args.dataset}/")[-1]
    fn = f"{root}/{folder}/{args.dataset}/{model}.pickle"

    if os.path.exists(fn):
        accuracies = pickle.load(open(fn, 'rb'))
    else:
        accuracies = dict()

    # Retrain per setup, store accuracies in dict, save dict as pickle
    seed = 42
    for retrain in retrain_configs:
        retrain = tuple([str(k) for k in retrain])
        if retrain in accuracies:
            logger.info(f"{retrain} already exists, continue")
            continue
        logger.info(f"Retraining setup {retrain}")
        train_acc, valid_noisy, valid_clean = main(args, seed, logger, retrain)
        accuracies[retrain] = train_acc, valid_noisy, valid_clean
        pickle.dump(accuracies, open(fn, "wb"))
