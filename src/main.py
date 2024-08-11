import torch
import random
import copy
import numpy as np
import pickle
import os
import transformers
from data import preprocess_dataset
from functools import partial
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from utils import (
    set_seed,
    validate,
    collate_tokenize,
    get_logger,
    get_args,
    get_model_family,
    get_tokenizer,
)


def find_freeze(n, freeze, model_name):
    """
    Return whether or not a parameter should be frozen.
    Args:
        - n (str): parameter name
        - freeze (list): list of layer numbers + "embeddings" to freeze
        - model_name (str): use model name to postprocess parametr names
    Returns:
        boolean
    """
    if "classifier" in n:
        return False
    if ("final_layer_norm" in n or "ln_f" in n.split(".")[1]) and not "layers" in n:
        return False
    if "embed" in n or "wpe" in n or "wte" in n or "project_in" in n:
        return "embeddings" in freeze
    if "bert" in model_name.lower():
        component = n.split(".")[1]
        if component in ["embeddings", "pooler"]:
            return component in freeze
        elif component in ["encoder", "decoder"]:
            layer = n.split(".")[3]
            return layer in freeze
    elif "opt" in model_name.lower():
        if "layers" in n.split(".")[2]:
            return n.split(".")[3] in freeze
        if "project_out" in n.split(".")[2]:
            return False
    elif "pythia" in model_name.lower():
        if "layers" in n.split(".")[1]:
            return n.split(".")[2] in freeze
    elif "neo" in model_name.lower():
        if "h" == n.split(".")[1]:
            return n.split(".")[2] in freeze
    raise NotImplementedError(n)


def get_optimizer(logger, model, freeze, lr, use_aux):
    """
    Get optimizer and optimizer for aux task.
    Args:
        - model
        - freeze: list of elements not to be modified by main task
        - lr (float): learning rate
        - use_aux (boolean): whether or not to create the auxiliary optimizer
    Returns:
        - AdamW optimizer
        - None or AdamW optimizer
    """
    names, params = zip(
        *[
            (n, p)
            for n, p in model.named_parameters()
            if not find_freeze(n, freeze, model.base)
        ]
    )
    logger.info("Trainable parameters for main task: " + str(names))
    optimizer = AdamW(params, lr=lr)

    # Auxiliary task can change all model parameters
    if use_aux:
        names, params = zip(
            *[
                (n, p)
                for n, p in model.named_parameters()
                if not find_freeze(n, ["embeddings"], model.base)
            ]
        )
        logger.info("Trainable parameters for auxiliary task: " + str(names))
        aux_optimizer = AdamW(params, lr=5e-6)
    else:
        aux_optimizer = None
    return optimizer, aux_optimizer


def get_scheduler(num_epochs, train, aux_train, optimizer, aux_optimizer):
    """
    Get learning rate scheduler for main task & aux task.
    Args:
        - num_epochs (int)
        - train: training examples DataLoader
        - aux_train: auxiliary task training examples DataLoader
        - optimizer: AdamW optimizer for the main task
        - aux_optimizer: AdamW optimizer for the auxiliary task
    Returns:
        - linear transformers lr scheduler for main task
        - None / linear transformers lr scheduler for main task
    """
    # Learning rate schedule
    num_training_steps1 = num_epochs * len(train)
    aux_lr_scheduler = None
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps1),
        num_training_steps=num_training_steps1,
    )
    if aux_train is not None:
        num_training_steps2 = num_epochs * len(aux_train)
        aux_lr_scheduler = transformers.get_scheduler(
            name="linear",
            optimizer=aux_optimizer,
            num_warmup_steps=int(0.1 * num_training_steps2),
            num_training_steps=num_training_steps2,
        )
    return lr_scheduler, aux_lr_scheduler


def report(logger, epoch, perf1, perf2):
    """
    Summarise performance in a particular epoch to user.
    Args:
        - logger: logging object
        - epoch: int
        - perf1: dict with accuracy and f1 keys
        - perf2: dict with accuracy and f1 keys
    """
    logger.info(
        f"Epoch {epoch}, "
        + f"train acc = {perf1['accuracy']:.3f}, f1 = {perf1['f1']:.3f}, "
        + f"validation acc = {perf2['accuracy']:.3f}, f1 = {perf2['f1']:.3f}"
    )


def train(
    logger,
    model,
    device,
    train_dataloader,
    eval_dataloader,
    num_epochs,
    lr,
    store=False,
    aux_eval_dataloader=None,
    aux_train_dataloader=None,
    freeze=[],
):
    """
    Train and evaluat with the main task and potentially an auxiliary one.
    Freeze all parameters contained in `freeze'.
    Args:
        - logger
        - model: HF model
        - device: cpu/gpu device
        - train_dataloader, eval_dataloader: DataLoader objects
        - num_epochs (int): how long to train for
        - lr (float): learning rate
        - store (boolean): whether or not to save models to disk
        - aux_eval_dataloader: None / DataLoader object
        - aux_train_dataloader: None / DataLoader object
        - freeze: list of strings indicating what to freeze e.g. \
            ["embeddings", '1', '2']
    """
    optimizer, aux_optimizer = get_optimizer(
        logger, model, freeze, lr, aux_train_dataloader is not None
    )
    lr_scheduler, aux_lr_scheduler = get_scheduler(
        num_epochs, train_dataloader, aux_train_dataloader, optimizer, aux_optimizer
    )

    # Train
    model.trace = dict()
    saved_memorised = False
    for epoch in range(num_epochs):
        losses = []

        # Mix main and auxiliary task
        dataloader = [(True, b) for b in train_dataloader]
        if aux_train_dataloader is not None:
            dataloader.extend([(False, b) for b in aux_train_dataloader])
        random.shuffle(dataloader)

        for use_main, batch in dataloader:
            model.train()
            batch = {k: v.to(device) for k, v in batch.items() if k != "idxs"}
            outputs = model(**batch, auxiliary=not use_main)
            losses.append(outputs.loss.item())
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Depending on task, use regular optimiser or auxiliary optimiser
            tmp_optimizer = optimizer if use_main else aux_optimizer
            tmp_scheduler = lr_scheduler if use_main else aux_lr_scheduler
            tmp_optimizer.step()
            tmp_scheduler.step()
            model.zero_grad()

        model.eval()
        # Evaluate on auxiliary task
        if aux_train_dataloader is not None:
            p1, _ = validate(aux_train_dataloader, model, device, True)
            p2, _ = validate(aux_eval_dataloader, model, device, True)
            report(logger, epoch, p1, p2)
            model.trace[epoch, "aux_train"] = p1
            model.trace[epoch, "aux_validation"] = p2

        # Evaluate on main task
        p1, _ = validate(train_dataloader, model, device)
        p2, _ = validate(eval_dataloader, model, device)
        report(logger, epoch, p1, p2)
        model.trace[epoch, "train"] = p1
        model.trace[epoch, "validation"] = p2

        logger.info(f"End of epoch. Training loss: {np.mean(losses):.3f}")
        if store:
            statedict = model.state_dict()
            if epoch + 1 == 50:
                torch.save((statedict, model.trace),
                           f"{model.name}_epoch={epoch+1}.pt")
            elif p1["f1"] >= 0.993 and not saved_memorised:
                torch.save((statedict, model.trace),
                           f"{model.name}_epoch=memorised.pt")
                saved_memorised = True

    perf, _ = validate(eval_dataloader, model, device)
    logger.info(
        f"End of training acc = {perf['accuracy']:.3f} f1 = {perf['f1']:.3f}")


def main(args, dataset, seed, logger, mmap=False, aux_dataset=None):
    """
    Model training for memorisation localisation.
    1 - performs dataset prep by creating DataLoader objects
    2 - calls the train function
    3 - calls final evaluation post training
    Args:
        - args: ArgumentParser args
        - dataset: dict w/ train/validation/test keys
        - seed: int
        - logger: logging obj
        - mmap: bool indicating whether or not we're making a memorisation map
          i.e. computing generalisation scores (see centroid analysis section)
        - aux_dataset: dict w/ train/validation/test keys
    Only returns something if mmap is True:
        - indices of datapoints that were in the training set
        - indices of datapoints that were in the test set
        - output logits for datapoints that were in the training set
        - output logits for datapoints that were in the test set
    """
    set_seed(seed)
    tokenizer = get_tokenizer(args.model_name)
    cfn = partial(collate_tokenize, tok=tokenizer)

    if mmap:  # For memorisation map: train on half, test on half
        idx = list(range(len(dataset["train"])))
        random.shuffle(idx)
        midpoint = int(len(idx) * 0.5)
        train_examples = [dataset["train"][i] for i in idx[:midpoint]]
        valid_examples = dataset["validation"]
        test_examples = [dataset["train"][i] for i in idx[midpoint:]]
    else:
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]
        test_examples = dataset["test"]

    # If there is an auxiliary task, use separate dataloaders
    if aux_dataset is not None:
        aux_train_dataloader = DataLoader(
            aux_dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=cfn,
        )
        aux_val_dataloader = DataLoader(
            aux_dataset["validation"], batch_size=32, collate_fn=cfn
        )
        aux_test_dataloader = DataLoader(
            aux_dataset["test"], batch_size=32, collate_fn=cfn
        )
    else:
        aux_val_dataloader, aux_test_dataloader, aux_train_dataloader = None, None, None

    # Turn dataset into batched data loaders
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size, collate_fn=cfn
    )
    train_dataloader_unshuf = DataLoader(
        train_examples, shuffle=False, batch_size=args.batch_size, collate_fn=cfn
    )
    val_dataloader = DataLoader(valid_examples, batch_size=32, collate_fn=cfn)
    test_dataloader = DataLoader(test_examples, batch_size=32, collate_fn=cfn)

    # Initialise model
    config = AutoConfig.from_pretrained(args.model_name.replace("_", "/"))
    config.pad_token_id = config.eos_token_id
    config.learn_subspaces = False  # Unused in paper
    config.cog_training = False  # Unused in paper
    config.num_labels = len(set([e["labels"] for e in dataset["train"]]))
    logger.info(f"{config.num_labels} labels in this dataset")
    config.num_labels2 = (
        0
        if aux_dataset is None
        else len(set([e["labels"] for e in aux_dataset["train"]]))
    )

    model = get_model_family(args.model_name).from_pretrained(
        args.model_name.replace("_", "/"), config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if (
        args.freeze
        == ["embeddings", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        and aux_dataset is None
    ):
        n = "-fullfreeze"
    else:
        n = ""
    model.name = (
        f"{args.checkpoint_folder}/{args.dataset}/{args.model_name}"
        + f"_seed={seed}_freeze={'-'.join(args.freeze) + n}"
        + f"{'_normal' if args.normal else ''}"
    )
    logger.info(model.name)
    if os.path.exists(f"{model.name}_epoch=50.pt") and not args.overwrite and not mmap:
        logger.info("This model already exists, exiting now...")
        exit()

    model.base = args.model_name

    train(
        logger,
        model,
        device,
        train_dataloader,
        val_dataloader,
        args.num_epochs,
        args.lr,
        aux_train_dataloader=aux_train_dataloader,
        aux_eval_dataloader=aux_val_dataloader,
        freeze=args.freeze,
        store=not mmap,
    )

    # Final evaluation post training
    p1, train_logits = validate(train_dataloader_unshuf, model, device)
    p2, test_logits = validate(test_dataloader, model, device)
    report(logger, "final", p1, p2)
    model.trace["final", "final_train"] = p1
    model.trace["final", "final_test"] = p2
    if aux_dataset is not None:
        p1 = validate(aux_train_dataloader, model, device, auxiliary=True)[0]
        p2 = validate(aux_test_dataloader, model, device, auxiliary=True)[0]
        report(logger, "final", p1, p2)
        model.trace["final", "final_aux_train"] = p1
        model.trace["final", "final_aux_test"] = p2

    if mmap:
        return idx[:midpoint], idx[midpoint:], train_logits, test_logits


if __name__ == "__main__":
    logger = get_logger()
    args = get_args(logger)

    # Initialise perturbed dataset
    dataset, idxs = preprocess_dataset(
        args.dataset, args.data_seed, normal=args.normal)
    if "noisy" in idxs:
        logger.info(f"{len(idxs['noisy'])} noisy examples")
        logger.info("First 25 noisy examples: "
                    + str(list(idxs["noisy"])[:25]))

    print(len(dataset["train"]))
    exit()

    # We can train in 3 ways:
    # 1. Create a memorisation map: train 30 models and vary whether examples
    # are train / test examples to get their training memorisation and
    # generalisation scores
    if args.training_type == "mmap":
        memorisation_map_data = dict()
        memorisation_map_data["idxs"] = idxs
        for seed in range(30):
            memorisation_map_data[seed] = main(
                args, copy.deepcopy(dataset), seed, logger, mmap=True
            )
        pickle.dump(
            memorisation_map_data,
            open(
                f"{args.checkpoint_folder}/mmap_{args.dataset}_{args.model_name}.pickle",
                "wb",
            ),
        )

    # 2. We train with a main and an auxiliary dataset while freezing certain
    # layers. We use this for the control experiment
    elif args.training_type == "freeze_train":
        aux_dataset, _ = preprocess_dataset(args.aux_dataset, normal=True)
        main(args, dataset, args.model_seed, logger, aux_dataset=aux_dataset)

    # 3. Or we "regularly" train on one task
    else:
        main(args, dataset, args.model_seed, logger)
