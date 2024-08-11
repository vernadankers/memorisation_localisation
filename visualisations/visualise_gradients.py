import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from visualisation_utils import VisualisationUtils
import numpy as np
import os
sns.set_context("talk")


def process_gradients(model, per_layer, freeze, modeln,
                      normalise_by_clean, norm):
    """
    Take a dictionary with weight name --> gradient mappings, and compute norms
    Args:
        - model: dictionary, keys are noisy/clean, values are weight-->grad
        - per_layer: dictionary to add results to
        - freeze (str): overview of frozen layers in this model
        - modeln (str): short model descriptor, e.g. BERT
        - normalise_by_clean (bool): whether to subtract clean examples' grads
        - norm (str): "l2" | "l1"
    Returns:
        dictionary mapping (layer_num, freeze str) to float, l1/l2 norm
    """
    for k1, v1 in model["noisy"].items():
        if "embed" in k1 or "final" in k1 or "wte" in k1 or "wpe" in k1 or "ln_f" in k1 or "pooler" in k1 or "classifier" in k1:
            continue

        # Subtract gradients from the clean examples
        if normalise_by_clean:
            v1 = v1 - model["clean"][k1]
        summed = torch.abs(v1**(2 if norm == 'l2' else 1)).sum().item()

        # Get the layer number
        if modeln in ["GPT-N", "Pythia"]:
            if len(k1.split('.')) < 3:
                continue
            else:
                layer_num = int(k1.split('.')[2])
        else:
            layer_num = int(k1.split('.')[3])
        per_layer[layer_num, freeze].append(summed)
    return per_layer


def get_per_layer(dataset, modeln, model, seed, freeze, epoch, normalise_by_clean, normalise_by_frozen, norm):
    """
    Create dict with gradient norms per layer.
    Args:
        - dataset (str): name of dataset, e.g. wic
        - modeln (str): shortened model name, e.g. BERT
        - model (str): full model name, e.g. bert-base-cased
        - seed (int): seed of the base model to analyse
        - freeze (str): string indicating which layers are frozen in model
        - epoch (int): 50 | memorised
        - normalise_by_clean (bool): whether to subtract clean example grads
        - normalise_by_frozen (bool): whether to normalise by frozen model grad
        - norm (str): "l1" | "l2"
    Returns:
        dict with layer numbers as keys, normalised grad magnitudes as values
    """
    per_layer = defaultdict(list)
    fn1 = f"../results/gradients/{dataset}/{model}_seed={seed}_freeze={freeze}_epoch={epoch}.pt.pickle"
    if freeze == "embeddings":
        fn2 = f"../results/gradients/{dataset}/{model}_seed=1_freeze=embeddings-0-1-2-3-4-5-6-7-8-9-10-11-fullfreeze_epoch=50.pt.pickle"
    else:
        fn2 = f"../results/gradients/{dataset}/{model}_seed=1_freeze=embeddings-0-1-2-3-4-5-6-7-8-9-10-11_epoch=50.pt.pickle"

    if not os.path.exists(fn1):
        print(f"missing... fn1: {fn1}")
    if not os.path.exists(fn2):
        print(f"missing... fn2: {fn2}")

    model = pickle.load(open(fn1, 'rb'))
    per_layer = process_gradients(
        model, per_layer, freeze, modeln, normalise_by_clean, norm)
    model = pickle.load(open(fn2, 'rb'))
    per_layer = process_gradients(
        model, per_layer, "embeddings-0-1-2-3-4-5-6-7-8-9-10-11", modeln,
        normalise_by_clean, norm)
    per_layer = {k: sum(v) for k, v in per_layer.items()}
    if norm == "l2":
        per_layer = {k: np.sqrt(v) for k, v in per_layer.items()}

    if normalise_by_frozen:
        for key, y in per_layer.items():
            if key[-1] == freeze:
                per_layer[key] = y / per_layer[tuple(key[:-1])+(
                    "embeddings-0-1-2-3-4-5-6-7-8-9-10-11",)]

    per_layer = {k: v for k, v in per_layer.items() if k[-1] == freeze}
    per_layer = {k: v/sum(per_layer.values()) for k, v in per_layer.items()}
    return per_layer


def get_extra_graphs():
    """
    Run gradients for models with frozen layers.
    Args:
        - ds_names: list of names of datasets to compute results for
        - normalise_by_clean (bool): whether to subtract clean examples' grads
        - noramlise_by_frozen (bool): whether to normalise by frozen grads
        - norm: "l1" | "l2"
    Returns:
        dictionary with accuracies per model, dataset combination
    """
    for mode in ["freeze", "clean", "noisy"]:
        for modeln, model in [("BERT", "bert-base-cased")]:
            allx, ally, allhue = [], [], []
            for dataset in ["mrpc", "trec"]:
                for focus_layers, freeze in utils.control_setups:
                    print(f"{modeln}, {dataset}, {freeze}")

                    per_layer = defaultdict(list)
                    if mode == "freeze":
                        fn1 = f"../results/gradients/{dataset}/{model}_seed=1_freeze=embeddings-0-1-2-3-4-5-6-7-8-9-10-11-fullfreeze_epoch=50.pt.pickle"
                        model_loaded = pickle.load(open(fn1, 'rb'))
                        for k1, v1 in model_loaded["clean"].items():
                            if "embed" in k1 or "final" in k1 or "wte" in k1 or "wpe" in k1 or "ln_f" in k1 or "pooler" in k1 or "classifier" in k1:
                                continue
                            summed = torch.abs(v1).sum().item()
                            # Get the layer number
                            if modeln in ["GPT-N", "Pythia"]:
                                if len(k1.split('.')) < 3:
                                    continue
                                else:
                                    layer_num = int(k1.split('.')[2])
                            else:
                                layer_num = int(k1.split('.')[3])
                            per_layer[layer_num, freeze].append(summed)
                        per_layer = {k: sum(v) for k, v in per_layer.items()}

                    else:
                        fn1 = f"../results/gradients/{dataset}/{model}_seed=1_freeze={freeze}_epoch=50.pt.pickle"
                        model_loaded = pickle.load(open(fn1, 'rb'))
                        for k1, v1 in model_loaded[mode].items():
                            if "embed" in k1 or "final" in k1 or "wte" in k1 or "wpe" in k1 or "ln_f" in k1 or "pooler" in k1 or "classifier" in k1:
                                continue
                            summed = torch.abs(v1).sum().item()
                            # Get the layer number
                            if modeln in ["GPT-N", "Pythia"]:
                                if len(k1.split('.')) < 3:
                                    continue
                                else:
                                    layer_num = int(k1.split('.')[2])
                            else:
                                layer_num = int(k1.split('.')[3])
                            per_layer[layer_num, freeze].append(summed)
                        per_layer = {k: sum(v) for k, v in per_layer.items()}

                    x, y = zip(*per_layer.items())
                    x, _ = zip(*x)
                    allx.extend(x)
                    ally.extend(y)
                    if mode == "freeze":
                        allhue.extend(
                            [f"all"]*len(x))
                    else:
                        allhue.extend(
                            [f"{focus_layers[0]+1}-{focus_layers[1]+1}"]*len(x))

            plt.figure(figsize=(4, 4))
            ax = sns.lineplot(x=allx, y=ally, hue=allhue,
                              palette="viridis", markers=True, style=allhue)
            sns.despine(top=True, right=True)
            plt.xlabel("layer")
            plt.ylabel("gradients")
            ax.set_xticks(range(0, 12), range(1, 13), fontsize=10)
            locs, labels = plt.yticks()
            ax.set_yticks(locs, labels, fontsize=10)
            plt.legend([], [], frameon=False)
            plt.savefig(f"gradients/extra_{mode}.pdf", bbox_inches="tight")


def control_setup(ds_names, normalise_by_clean, normalise_by_frozen, norm):
    """
    Run gradients for models with frozen layers.
    Args:
        - ds_names: list of names of datasets to compute results for
        - normalise_by_clean (bool): whether to subtract clean examples' grads
        - noramlise_by_frozen (bool): whether to normalise by frozen grads
        - norm: "l1" | "l2"
    Returns:
        dictionary with accuracies per model, dataset combination
    """
    scores = defaultdict(lambda: dict())
    for modeln, model in utils.model_setups:
        allx, ally, allhue = [], [], []
        for dataset in ds_names:
            accuracy1, accuracy2 = [], []
            for focus_layers, freeze in utils.control_setups:
                print(f"{modeln}, {dataset}, {freeze}")
                per_layer = get_per_layer(
                    dataset, modeln, model, 1, freeze, 50, normalise_by_clean,
                    normalise_by_frozen, norm)
                if per_layer is None:
                    continue

                x, y = zip(*per_layer.items())
                layer, _ = zip(*x)
                layers, _ = zip(
                    *Counter({x_: y_ for x_, y_ in zip(layer, y)}).most_common(2))
                for l in layers:
                    accuracy2.append(l in focus_layers)
                layers, _ = zip(
                    *Counter({x_: y_ for x_, y_ in zip(layer, y)}).most_common(1))
                for layer in layers:
                    accuracy1.append(layer in focus_layers)
                x, _ = zip(*x)
                allx.extend(x)
                ally.extend(y)
                allhue.extend(
                    [f"{focus_layers[0]+1}-{focus_layers[1]+1}"]*len(x))
            scores[modeln, dataset]["accuracy@1"] = np.mean(accuracy1)
            scores[modeln, dataset]["accuracy@2"] = np.mean(accuracy2)

        plt.figure(figsize=(4, 4))
        ax = sns.lineplot(x=allx, y=ally, hue=allhue,
                          palette="viridis", markers=True, style=allhue)
        sns.despine(top=True, right=True)
        plt.xlabel("layer")
        plt.ylabel("gradients")
        ax.set_xticks(range(0, 12), range(1, 13), fontsize=10)
        locs, labels = plt.yticks()
        ax.set_yticks(locs, labels, fontsize=10)
        if (normalise_by_frozen, normalise_by_clean) == (False, False):
            plt.legend(fontsize=11)
        else:
            plt.legend([], [], frameon=False)
        plt.savefig(
            f"validation_gradients_frozen={normalise_by_frozen}_clean"
            + f"={normalise_by_clean}_norm={norm}_model={modeln}.pdf",
            bbox_inches="tight")
        plt.show()
    return dict(scores)


def gradient_validation():
    ds_names = ["mrpc", "trec"]
    scores = defaultdict(lambda: dict())

    for norm in ['l2', 'l1']:
        for normalise_by_frozen in [False, True]:
            for normalise_by_clean in [False, True]:
                print(norm, normalise_by_clean, normalise_by_frozen)
                scores[normalise_by_clean, normalise_by_frozen, norm] = control_setup(
                    ds_names, normalise_by_clean, normalise_by_frozen, norm=norm)
    pickle.dump(dict(scores), open(
        "pickled_results/gradient_validation.pickle", 'wb'))


def gradient_testing():
    normalise_by_frozen, normalise_by_clean = True, True
    norm = 'l1'
    scores = control_setup(utils.control_setup_ds_names, normalise_by_clean,
                           normalise_by_frozen, norm)
    pickle.dump(dict(scores), open(
        "pickled_results/gradient_testing.pickle", 'wb'))


def gradient_main():
    normalise_by_clean, normalise_by_frozen = True, True

    freeze = "embeddings"
    normalise_by_normal = False
    results = defaultdict(lambda: dict())
    summary = defaultdict(lambda: dict())
    for modeln, model in utils.model_setups:
        print(modeln)
        for setup, ds_names in utils.data_setups[:-1]:
            allx_, ally_, allh_ = [], [], []
            for dataset in ds_names:
                allx, ally = [], []
                for seed in [1, 2, 3]:
                    if not os.path.exists(f"../results/gradients/{dataset}/{model}_seed={seed}_freeze={freeze}_epoch=50.pt.pickle"):
                        print(dataset, seed)
                        continue
                    print(f"    - {dataset}")

                    per_layer = get_per_layer(
                        dataset, modeln, model, seed, freeze, "50",
                        normalise_by_clean, normalise_by_frozen, norm='l1')

                    x, y = zip(*per_layer.items())
                    x, _ = zip(*x)
                    allx.append(x)
                    ally.append(y)

                allx = np.mean(allx, axis=0)
                ally = np.mean(ally, axis=0)
                allx_.extend(allx)
                ally_.extend(ally)
                allh_.extend([dataset]*len(allx))

                summary[model][dataset] = np.sum(
                    [x_*y_ for x_, y_ in zip(range(0, 12), ally)])/np.sum(ally)
                results[model][dataset] = ally

            sns.set_context("talk")
            plt.figure(figsize=(4, 1.7))
            sns.set_style("white")
            plt.grid(axis='y', zorder=-1)
            ax = sns.lineplot(x=allx_, y=ally_, hue=allh_, palette=utils.palette,
                              style=allh_, zorder=1, linewidth=4, errorbar=None)

            # Only include scatterplot for the final layer
            ax = sns.scatterplot(
                x=[y for y in allx_ if y == 11],
                y=[y for y, z in zip(ally_, allx_) if z == 11],
                hue=[y for y, z in zip(allh_, allx_) if z == 11],
                palette=utils.palette, style=[
                    y for y, z in zip(allh_, allx_) if z == 11],
                markers=utils.marker_dict, edgecolor='black', s=200, alpha=0.8)
            plt.legend([], [], frameon=False)
            plt.xlim(-0.5, 11.5)
            plt.xlabel("")
            ax.set_xticks([])
            plt.ylim(0, 0.2)
            plt.yticks([0.05, 0.1, 0.15])
            if setup == "nlu":
                locs, labels = plt.yticks()
                plt.ylabel(f"gradients\npostprocessed", fontsize=12)
                ax.set_yticks(locs, labels, fontsize=12)
            else:
                plt.ylabel("")
                ax.set_yticks(locs, [])
            plt.ylim(0, 0.2)
            sns.despine(top=True, right=True)
            plt.savefig(
                f"gradients/gradients_{setup}_{modeln}.pdf", bbox_inches="tight")
            plt.show()
    pickle.dump((dict(results), dict(summary)), open(
        "pickled_results/gradients_main.pickle", 'wb'))


def summarise_gradient_validation():
    res = pickle.load(open("pickled_results/gradient_validation.pickle", 'rb'))
    for normalise_by_clean, normalise_by_frozen in [
            (False, False), (False, True), (True, False), (True, True)]:
        line1 = ""
        for model in ["Pythia", "GPT-N", "BERT", "OPT"]:
            acc1 = (res[normalise_by_clean, normalise_by_frozen, "l1"][model, "trec"]["accuracy@2"]
                    + res[normalise_by_clean, normalise_by_frozen, "l1"][model, "mrpc"]["accuracy@2"])/2
            acc2 = (res[normalise_by_clean, normalise_by_frozen, "l2"][model, "trec"]["accuracy@2"]
                    + res[normalise_by_clean, normalise_by_frozen, "l2"][model, "mrpc"]["accuracy@2"])/2
            line1 += f"{acc1:.2f} & {acc2:.2f} & "
        print(line1 + "\\\\")


if __name__ == "__main__":
    utils = VisualisationUtils()
    #get_extra_graphs()
    #gradient_validation()
    #summarise_gradient_validation()
    #gradient_testing()
    gradient_main()
