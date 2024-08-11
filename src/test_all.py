import torch
import pickle
from data import preprocess_dataset
from functools import partial
from torch.utils.data import DataLoader
from utils import (
    set_seed,
    validate,
    collate_tokenize,
    get_logger,
    get_model,
    get_tokenizer,
)


def main(model_name, custom_model, dataset, seed, logger):
    set_seed(seed)
    tokenizer = get_tokenizer(model_name)
    cfn = partial(collate_tokenize, tok=tokenizer)
    train_dataloader = DataLoader(
        dataset["train"], batch_size=16, collate_fn=cfn)
    val_dataloader = DataLoader(
        dataset["validation"], batch_size=16, collate_fn=cfn)
    test_dataloader = DataLoader(
        dataset["test"], batch_size=16, collate_fn=cfn)

    # Initialise model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, custom_model)
    model.to(device)

    # Final evaluation post training
    p1, _ = validate(train_dataloader, model, device)
    p2, _ = validate(val_dataloader, model, device)
    p3, _ = validate(test_dataloader, model, device)

    logger.info(
        f"train acc = {p1['accuracy']:.3f}, f1 = {p1['f1']:.3f}, "
        + f"validation acc = {p2['accuracy']:.3f}, f1 = {p2['f1']:.3f}"
        + f"test acc = {p3['accuracy']:.3f}, f1 = {p3['f1']:.3f}"
    )
    return p1, p2, p3


if __name__ == "__main__":
    logger = get_logger()
    results = dict()
    for dataset_name in ["wic", "boolq", "cola", "sst2", "sst5", "emotion",
                         "implicithate", "stormfront", "reuters"]:
        for model in ["EleutherAI_gpt-neo-125m"]:
            # freeze = "embeddings"
            # for seed in [1, 2, 3]:
            #     logger.info(f"{dataset_name}, {model}, {seed}, {freeze}")
            #     custom_model = f"/home/s2112866/memorisation_localisation/checkpoints/{dataset_name}/{model}_seed={seed}_freeze={freeze}_epoch=50.pt"
            #     # Initialise perturbed dataset
            #     dataset, idxs = preprocess_dataset(
            #         dataset_name, 1)
            #     results[seed, dataset_name, model, freeze] = main(
            #         model, custom_model, dataset, seed, logger)

            seed = 1
            for freeze in ["embeddings-2-3-4-5-6-7-8-9-10-11",
                           "embeddings-0-1-2-3-4-7-8-9-10-11",
                           "embeddings-0-1-2-3-4-5-6-7-8-9"]:
                logger.info(f"{dataset_name}, {model}, {seed}, {freeze}")
                custom_model = f"/home/s2112866/reproductions/memorisation_localisation/checkpoints/{dataset_name}/{model}_seed={seed}_freeze={freeze}_epoch=50.pt"
                # Initialise perturbed dataset
                dataset, idxs = preprocess_dataset(
                    dataset_name, 1, normal=False)
                try:
                    results[seed, dataset_name, model, freeze] = main(
                        model, custom_model, dataset, seed, logger)
                except:
                    results[seed, dataset_name, model, freeze] = None
                    continue

    pickle.dump(results, open("results_reproductions.pickle", 'wb'))
