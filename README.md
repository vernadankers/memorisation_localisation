# Memorisation localisation for natural language classification tasks

0. Prepare by installing the conda environment using `conda env create --file=env.yaml`, which will install the environment `memloc` in python 3.9.
1. Add folders `logs`, `checkpoints` and subfolders to the root folder by running `bash setup_folders.sh`.
2. Train models by running `run_all.sh training` from within `src/submit_scripts/`. Model checkpoints will be stored to the `checkpoints/<dataset>` folder, and during training / analysis progress information will be saved to `logs/<analysis_type>`.
3. Subsequently, analyses can be conducted by running `run_all.sh <analysis>` from within `src/submit_scripts` where analysis is one of `swapping | retraining | gradients | probing | centroid_analysis`.
4. Individual analyses can be visualised using the corresponding notebooks (`visualise_layer_swapping.ipynb`, `visualise_layer_retraining.ipynb`, `visualise_gradients.py`, `visualise_probing.ipynb`), that start with cells for the control setup (section 3.2), followed by cells the main results analysis (section 4).
5. Centroid analysis can be performed using `visualise_centroid_analysis.ipynb`, after first executing `visualise_mmaps.ipynb` for all models / datasets, to compute the generalisation scores used in the centroid analysis correlation analysis.
6. Afterwards, summary visualisations can be computed using `summarising_visualisations.ipynb`.
7. For the appendix experiments using the 1.3B models, execute `run_all.sh <mode>` from within `src/submit_scripts_big/` first using `training`, followed by `swapping` and `centroid_analysis`.
