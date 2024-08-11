import torch
import copy
import pickle
import random
import os
from data import preprocess_dataset
from torch.utils.data import DataLoader
from functools import partial
from utils import collate_tokenize, validate, set_seed, get_logger, get_args, \
    get_model, get_tokenizer


def main(args, tokenizer, model, dataset, idxs):
    """
    Store forgetting gradients for noisy & clean examples to pickled file.
    Args:
        - args: argparse object
        - tokenizer: HF AutoTokenizer instantiation
        - model: HF model class instantiation
        - dataset: datasets.Dataset object, w/ train, validation & test keys
        - idxs: dict mapping noisy&clean to list of indices for train examples
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.model_seed)
    model.to(device)
    model.eval()

    clean_idx = list(idxs["clean"])
    random.shuffle(clean_idx)
    focus_groups = {
        "noisy": idxs["noisy"],
        "clean": clean_idx[:len(idxs["noisy"])]}

    all_gradients = dict()
    for group_name in focus_groups:
        subset = [dataset["train"][i] for i in focus_groups[group_name]]
        train_dataloader = DataLoader(
            subset, shuffle=False, batch_size=1,
            collate_fn=partial(collate_tokenize, tok=tokenizer))

        # perf = validate(train_dataloader, model, device)[0]
        # logger.info(
        #     f"Training acc = {perf['accuracy']:.3f}, f1 = {perf['f1']:.3f}")

        gradients_forget = dict()
        for batch in train_dataloader:
            try:
                batch = {k: v.to(device)
                         for k, v in batch.items() if k != 'idxs'}

                # Backprop the negative loss
                outputs = model(**batch)
                loss = torch.nn.NLLLoss()
                loss = -1 * loss(outputs.logits, batch["labels"])
                loss.backward()

                for k, v in model.named_parameters():
                    if v.grad is None:
                        continue
                    if not ("embed" in k or "wpe" in k or "wte" in k or "project_in" in k or "pooler" in k):
                        if k in gradients_forget:
                            gradients_forget[k] += copy.deepcopy(
                                v.grad.data).to("cpu")
                        else:
                            gradients_forget[k] = copy.deepcopy(
                                v.grad.data).to("cpu")

                model.zero_grad()
            except:
                continue

        # Average over datapoints
        for key in gradients_forget:
            gradients_forget[key] = gradients_forget[key] / len(subset)
        all_gradients[group_name] = gradients_forget

    folder = args.checkpoint_folder.replace('checkpoints', 'results')
    model = args.custom_model.split(f'checkpoints/{args.dataset}/')[-1]
    pickle.dump(all_gradients, open(
        f"{folder}/gradients/{args.dataset}/{model}.pickle", 'wb'))


if __name__ == "__main__":
    logger = get_logger()
    args = get_args(logger)
    set_seed(args.model_seed)
    dataset, idxs = preprocess_dataset(args.dataset, args.data_seed)
    logger.info(list(idxs["noisy"])[:25])
    tokenizer = get_tokenizer(args.model_name)
    if os.path.exists(args.custom_model.replace("epoch=50", "epoch=memorised")):
        model = get_model(
            args.model_name,
            args.custom_model.replace("epoch=50", "epoch=memorised"))
    else:
        model = get_model(args.model_name, args.custom_model)
    main(args, tokenizer, model, dataset, idxs)
