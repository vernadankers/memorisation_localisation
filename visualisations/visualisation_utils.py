import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class VisualisationUtils:
    def __init__(self):
        self.ds_names = [
            "wic", "rte", "mrpc", "cola", "boolq", "sst2", "sst5",
            "emotion",  "implicithate", "stormfront", "reuters", "trec"]

        self.control_setup_ds_names = [
            "wic", "cola", "boolq", "sst2", "sst5",
            "emotion",  "implicithate", "stormfront", "reuters"]

        self.control_setups = [
            ((0, 1), "embeddings-2-3-4-5-6-7-8-9-10-11"),
            ((5, 6), "embeddings-0-1-2-3-4-7-8-9-10-11"),
            ((10, 11), "embeddings-0-1-2-3-4-5-6-7-8-9")]

        self.data_setups = [("nlu", ["wic", "rte", "mrpc", "cola", "boolq"]),
                            ("hate", ["implicithate", "stormfront"]),
                            ("sentiment", ["sst2", "sst5", "emotion"]),
                            ("tc", ["trec", "reuters"]),
                            ("all", ["wic", "rte", "mrpc", "cola", "boolq", "implicithate",
                                     "stormfront", "sst2", "sst5", "emotion", "trec", "reuters"])]
        self.model_setups = [
             ("BERT", "bert-base-cased"),
             ("OPT", "facebook_opt-125m"),
             ("GPT-N", "EleutherAI_gpt-neo-125m"),
             ("Pythia", "EleutherAI_pythia-160m-deduped"),
        ]
        self.marker_dict = {
            n: m for n, m in zip(self.ds_names,
                                 ['o', 'X', 's', 'P', 'D', (4, 1, 0), '^',
                                  (4, 1, 45), 'v', (8, 1, 0), 'H', '*'])}
        self.palette = {n: m for n, m in zip(
            self.ds_names, sns.color_palette("Spectral", 12))}
        self.palette['sst5v2'] = self.palette['sst5']
        self.palette['trecv2'] = self.palette['trec']
        self.palette['emotionv2'] = self.palette['emotion']
        self.palette['reutersv2'] = self.palette['reuters']
        self.palette['implicithatev2'] = self.palette['implicithate']
        self.marker_dict['sst5v2'] = self.marker_dict['sst5']
        self.marker_dict['trecv2'] = self.marker_dict['trec']
        self.marker_dict['emotionv2'] = self.marker_dict['emotion']
        self.marker_dict['reutersv2'] = self.marker_dict['reuters']
        self.marker_dict['implicithatev2'] = self.marker_dict['implicithate']

    def save_heatmap(self, heatmap, filename, numlayers=12, label="swapped"):
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(
            heatmap,
            edgecolor='black',
            linewidth=0,
            cmap=sns.light_palette("seagreen", as_cmap=True),
            annot=numlayers == 12, annot_kws={'fontsize': 9}, fmt='.1f',
            cbar=False, vmin=0, vmax=1)
        if numlayers == 12:
            ax.set_xticks(
                [z + 0.5 for z in range(0, numlayers)],
                range(1, numlayers+1), fontsize=13)
            ax.set_yticks(
                [z + 0.5 for z in range(0, numlayers)],
                range(1, numlayers+1), fontsize=13, rotation=0)
        else:
            ax.set_xticks(
                [z + 0.5 for z in range(0, numlayers, 2)],
                range(1, numlayers+1, 2), fontsize=13)
            ax.set_yticks(
                [z + 0.5 for z in range(0, numlayers, 2)], range(
                    1, numlayers+1, 2), fontsize=13, rotation=0)

        ax.set_ylabel(f"#layers {label}")
        ax.set_xlabel("layers affected")
        ax.axhline(y=0, color='k', linewidth=3)
        ax.axhline(y=numlayers, color='k', linewidth=3)
        ax.axvline(x=0, color='k', linewidth=3)
        ax.axvline(x=numlayers, color='k', linewidth=3)
        plt.savefig(filename, bbox_inches="tight", transparent=True)
        plt.show()

    def save_legend(self):
        ds_names = [
            "WiC", "RTE", "MRPC", "CoLA", "BoolQ", "SST-2", "SST-5",
            "Emotion",  "ImplicitHate", "Stormfront", "Reuters", "TREC"]

        marker_dict = {
            n: m for n, m in zip(ds_names,
                                 ['o', 'X', 's', 'P', 'D', (4, 1, 0), '^',
                                  (4, 1, 45), 'v', (8, 1, 0), 'H', '*'])}
        palette = {n: m for n, m in zip(
            ds_names, sns.color_palette("Spectral", 12))}

        ax = sns.scatterplot(
            x=range(12), s=200, y=range(12), hue=ds_names,
            style=ds_names, palette=palette, markers=marker_dict)
        legend = plt.legend(bbox_to_anchor=(1, 1.05), frameon=False, ncols=4)

        def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            for ha in ax.legend_.legendHandles:
                ha.set_edgecolor("black")
            fig.savefig("legend.png", dpi="figure", bbox_inches=bbox)
            fig.savefig("legend2.pdf", dpi="figure", bbox_inches=bbox)

        export_legend(legend)
        plt.show()

    def get_heatmap_and_averages(self, filename, retraining=True, opt_big=False):
        if retraining and opt_big:
            raise NotImplementedError("Retraining not implement for OPT1.3b")
            exit()
        pickled_data = pickle.load(open(filename, 'rb'))
        aggregated_results = defaultdict(lambda: [])
        clean_accs = []
        for layer_combination in pickled_data:
            for l in layer_combination:
                if retraining:
                    accuracy = pickled_data[layer_combination][1]['accuracy']
                    min_accuracy = pickled_data[(
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11')][1]['accuracy']
                    accuracy = (accuracy - min_accuracy) / (1 - min_accuracy)
                else:
                    accuracy = pickled_data[layer_combination][0]
                    clean_accs.append(pickled_data[layer_combination][1])
                aggregated_results[len(layer_combination),
                                   int(l)].append(accuracy)

        num_layers = 12 if not opt_big else 24
        heatmap = np.zeros((num_layers, num_layers))
        for i, j in aggregated_results:
            # aggregated_results contains accuracy --> turn into error
            heatmap[i-1][j] = 1 - np.mean(aggregated_results[i, j])
        average_over_num_layers_affected = np.array(
            heatmap).mean(axis=0).tolist()
        if retraining:
            return heatmap, average_over_num_layers_affected
        return heatmap, average_over_num_layers_affected, clean_accs

    def visualise_single_row(self, results, retraining=True):
        plt.figure(figsize=(4, 4))
        ax = sns.swarmplot(
            x=[i for i, m in enumerate(results.keys()) for d in results[m]],
            y=[max(results[m][d]) for m in results for d in results[m]],
            color='grey')
        model_names, _ = zip(*self.model_setups)
        ax.set_xticks([0, 1, 2, 3], model_names)
        sns.despine()
        plt.legend([], [], frameon=False)

        if retraining:
            plt.ylabel("max memorisation \nerror retraining 1 layer")
            plt.savefig("layer_retraining/retraining_1layer.pdf",
                        bbox_inches='tight')
        else:
            plt.ylabel("max memorisation \nerror swapping 1 layer")
            plt.savefig("layer_swapping/swapping_1layer.pdf",
                        bbox_inches='tight')

    def visualise_flattened_heatmap(self, allx, ally, allh, setup, modeln,
                                    retraining=True, ymin=0.3, ymax=0.9):
        plt.figure(figsize=(4, 1.7))
        sns.set_style("white")
        plt.grid(axis='y', zorder=-1)
        ax = sns.lineplot(x=allx, y=ally, hue=allh, palette=self.palette, style=allh, zorder=1,
                          linewidth=3, errorbar=None)

        # Only scatter points for the final layer, to avoid visual clutter
        ax = sns.scatterplot(
                        x=[y for y, z in zip(allx, allx) if z == 11],
                        y=[y for y, z in zip(ally, allx) if z == 11],
                        hue=[y for y, z in zip(allh, allx) if z == 11],
                        palette=self.palette, style=[
                            y for y, z in zip(allh, allx) if z == 11],
                        markers=self.marker_dict, edgecolor='black', s=200, alpha=0.8)
        plt.legend([], [], frameon=False)
        plt.xlabel("")  # layer")
        plt.yticks([0.4, 0.6, 0.8])

        # Only include ylabel for NLU, we'll put the graphs next to each other
        plt.ylim(ymin, ymax)
        if setup == "nlu":
            plt.ylabel(f"memorisation\nerror", fontsize=12)
            locs, labels = plt.yticks()
            ax.set_yticks(locs, [f"{l:.2f}" for l in locs],
                          fontsize=12, rotation=0)
        else:
            ax.set_yticklabels([])

        ax.set_xticks([])  # range(0, 12), range(1, 13), fontsize=12)
        sns.despine(top=True, right=True)
        plt.xlim(-0.5, 11.5)
        if retraining:
            plt.savefig(
                f"layer_retraining/layer_retraining_{setup}_{modeln}.pdf", bbox_inches="tight")
        else:
            plt.savefig(
                f"layer_swapping/layer_swapping_{setup}_{modeln}.pdf", bbox_inches="tight")
        plt.show()

    def visualise_events(self, x, y, ds_names, filename, big=False, probing=False):
        plt.figure(figsize=(5, 5))
        sns.scatterplot(
                    x=x,
                    y=y,
                    style=ds_names,
                    hue=ds_names,
                    palette=self.palette, markers=self.marker_dict,
                    alpha=0.8, s=400, edgecolor='black')
        ax = sns.lineplot(x=[1, 12 if not big else 24],
                          y=[1, 12 if not big else 24], color="grey", linestyle='--', zorder=-1)
        plt.legend([], [], frameon=False)  # bbox_to_anchor=(1, 1.05))

        for a, b, d_ in zip(x, y, ds_names):
            c = self.palette[d_]
            sns.lineplot(x=[a, a+0.0001], y=[0, b],
                         color=c, linestyle='-', zorder=-1)
            sns.lineplot(x=[0, a+0.0001], y=[b, b],
                         color=c, linestyle='-', zorder=-1)

        if probing:
            plt.xlabel("memorisation >> generalisation (noisy)", fontsize=14.5)
            plt.ylabel(r"90% $F_1$ (clean)", fontsize=15)
        else:
            plt.xlabel("crossing")
            plt.ylabel("classification initiation")
        xticklabels = range(1, 25, 2) if big else range(1, 13)
        ax.set_xticks(xticklabels, xticklabels, fontsize=10)
        ax.set_yticks(xticklabels, xticklabels, fontsize=10)
        sns.despine(top=True, right=True)
        plt.legend([], [], frameon=False)
        plt.xlim(1, 12.5 if not big else 24.5)
        plt.ylim(1, 12.5 if not big else 24.5)
        plt.savefig(filename, bbox_inches="tight")
        plt.show()

    def shorten(self, name):
        return name.replace("facebook_opt-125m", "OPT").replace("EleutherAI_pythia-160m-deduped", "Pythia").replace(
            "EleutherAI_gpt-neo-125m", "GPT-N").replace("bert-base-cased", "BERT")

    def capitalise(self, name):
        return {"wic": "WiC", 'rte': "RTE", "mrpc": "MRPC", "cola": "CoLA",
                "boolq": "BoolQ", "sst2": "SST-2", "sst5": "SST-5",
                "emotion": "Emotion", "implicithate": "ImplicitHate",
                "stormfront": "Stormfront", "reuters": "Reuters",
                "trec": "TREC"}[name]

    def cog(self, weights):
        sum_ = 0
        for l, w in zip(range(1, 13), weights):
            sum_ += l*w
        return sum_ / sum(weights)

    def minmax(self, weights):
        mini = min(weights)
        maxi = max(weights)
        return [(x-mini)/(maxi - mini) for x in weights]

    def plot_lines(self, ax):
        ax.vlines(0, 0, 3, color='black', linestyle='-', linewidth=4)
        ax.vlines(1, 0, 1, color='black', linestyle='-')
        ax.vlines(2, 1, 2, color='black', linestyle='-')
        ax.vlines(3, 2, 3, color='black', linestyle='-', linewidth=4)
        ax.hlines(3, 0, 3, color='black', linestyle='-', linewidth=4)
        ax.hlines(0, 0, 1, color='black', linestyle='-')
        ax.hlines(1, 1, 2, color='black', linestyle='-')
        ax.hlines(2, 2, 3, color='black', linestyle='-')


if __name__ == "__main__":
    utils = VisualisationUtils()
    utils.save_legend()
