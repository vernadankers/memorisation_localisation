import numpy as np
import torch
import argparse
import pickle
from collections import defaultdict
from utils import collate_tokenize, get_model, get_tokenizer, get_logger
from data import preprocess_dataset
from layer_swapping import mix as mix_models
from skspatial.objects import Line


def get_representations(layer, dataset1, dataset2, idxs, model, tokenizer,
                        focus_label, other_label, modeltype):
    """
    Get representations of examples of class focus_label.
    Only retrieve noisy reps for examples that come from other_label,
    to isolate label swap for two specific classes.
    Args:
        - layer (int): layer currently under investigation
        - dataset1: noisy dataset
        - dataset2: clean dataset
        - idxs: dict with info about which indices are noisy
        - model: model to collect reps from
        - tokenizer: tokenizer used to encode inputs
        - focus_label (int): class that we're collecting reps from
        - other_label (int): original class for noisy examples
    Returns:
        dict with representations per example
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    reps = dict()
    for example in dataset1["train"]:
        i = example["idx"]
        # Focus on focus_label examples
        if example["labels"] != focus_label:
            continue
        # Only retrieve noisy examples that come from other_label
        if i in idxs["noisy"] and dataset2["train"][i]["labels"] != other_label:
            continue
        inputs = collate_tokenize([example], tokenizer)
        # Catch exceptional cases with empty inputs
        if inputs["input_ids"].shape[-1] == 0:
            continue
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "idxs"}
        example_reps = model(**inputs)[-1]
        example_reps = tuple(x.detach().cpu() for x in example_reps)[
                             layer][0][0 if "bert" in modeltype else -1].flatten()
        reps[i] = {"id": i, "vec": example_reps.tolist()}
    return reps


def get_indices_and_representations(model_name, layer, custom_model1, custom_model2, dataset_name, mix=False, class1=0, class2=1):
    dataset, idxs = preprocess_dataset(dataset_name,  1)
    dataset_clean, _ = preprocess_dataset(dataset_name,  1, normal=True)

    # Load tokenizer, and model
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, custom_model1)
    model_state_dict = torch.load(
        custom_model1, map_location=torch.device('cpu'))[0]
    model.load_state_dict(model_state_dict, strict=False)

    # Modify model when mixing top / bottom
    if mix:
        model_state_dict2 = torch.load(
            custom_model2, map_location=torch.device('cpu'))[0]
        if mix == "mixb":
            model.load_state_dict(
                mix_models(model_state_dict, model_state_dict2, [0, 1, 2, 3, 4, 5], model.base), strict=False)
        elif mix == "mixt":
            model.load_state_dict(
                mix_models(model_state_dict, model_state_dict2, [6, 7, 8, 9, 10, 11], model.base), strict=False)

    reps1 = get_representations(layer, dataset, dataset_clean, idxs,
                                model, tokenizer, class1, class2, model_name)
    reps2 = get_representations(layer, dataset, dataset_clean, idxs,
                                model, tokenizer, class2, class1, model_name)
    return idxs, reps1, reps2


def get_centroid(reps, idxs):
    return np.mean(
        np.array([v["vec"] for v in reps.values()
                  if v["id"] not in idxs["noisy"]]), axis=0).tolist()


def main(args):
    layers = defaultdict(lambda: list())
    distances = defaultdict(lambda: list())
    folder = args.checkpoint_folder.replace(
        "checkpoints", "results")
    mix = args.setup
    pickled_name = f"{folder}/centroid_analysis/{args.dataset}/{args.model_name}_{mix}.pickle"

    for seed in [1, 2, 3]:
        model1 = f"{args.checkpoint_folder}/{args.dataset}/{args.model_name}_seed={seed}_freeze=embeddings_epoch=50.pt"
        model2 = f"{args.checkpoint_folder}/{args.dataset}/{args.model_name}_seed={seed}_freeze=embeddings_normal_epoch=50.pt"

        class_combinations = [
            (c1, c2) for c1 in range(args.num_labels)
            for c2 in range(args.num_labels) if c1 != c2]

        for c1, c2 in class_combinations:
            for layer in range(1, 13 if "1.3" not in args.model_name else 25):
                logger.info(f"------------ LAYER {layer} ------------")
                idxs, reps1, reps2 = get_indices_and_representations(
                    args.model_name, layer, model1, model2, args.dataset,
                    mix, c1, c2)

                # Find the centroid of both classes
                point1 = get_centroid(reps1, idxs)
                point2 = get_centroid(reps2, idxs)

                # Get line with point1 at the origin, moving through point2
                line = Line.from_points(point_a=np.array(
                    point1), point_b=np.array(point2))
                pointb = line.transform_points([point2])[0]

                # Project all points on the line
                dists1 = line.transform_points(
                    [v["vec"] for v in reps1.values()])
                dists2 = line.transform_points(
                    [v["vec"] for v in reps2.values()])

                # Normalise by the distance to pointb
                dists2 = [x/pointb for x, v in zip(
                    dists2, reps2.values()) if v["id"] not in idxs["noisy"]]
                noisy_dists = [x/pointb for x, v in zip(
                    dists1, reps1.values()) if v["id"] in idxs["noisy"]]
                dists1 = [x/pointb for x, v in zip(
                    dists1, reps1.values()) if v["id"] not in idxs["noisy"]]
                distances["class1"].extend(dists1)
                distances["class2"].extend(dists2)
                distances["noisy"].extend(noisy_dists)
                layers["class1"].extend([layer-1]*len(dists1))
                layers["class2"].extend([layer-1]*len(dists2))
                layers["noisy"].extend([layer-1]*len(noisy_dists))

                pickle.dump((dict(layers), dict(distances)),
                            open(pickled_name, 'wb'))


if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--setup", default="memorised")
    parser.add_argument("--checkpoint_folder", type=str,
                        default="/home/s2112866/reproductions/memorisation_localisation/checkpoints")
    args = parser.parse_args()

    main(args)
