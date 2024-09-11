#!/bin/bash
# this script runs all hyperparameter tuning scripts
ZERO='[----]'
ONE='[#---]'
TWO='[##--]'
THREE='[###-]'
FOUR='[####]'

# run hyperparameter tuning for lorenz
function pbar {
    clear
    echo $1
}

pbar $ZERO
echo "Running PCA hyperparameter tuning"
conda activate maco_rev1
python genres_pca_htune.py
conda deactivate

pbar $ONE
echo "Running ICA hyperparameter tuning"
conda activate maco_rev1
python genres_ica_htune.py
conda deactivate

pbar $TWO
echo "Running DCA hyperparameter tuning"
conda activate dca
python genres_dca_htune.py
conda deactivate

pbar $THREE
echo "Running SFA hyperparameter tuning"
conda activate maco_rev1
python genres_sfa_htune.py
conda deactivate

pbar $FOUR
echo "Create unified figure"
conda activate maco_rev1
python genres_final_htune.py
conda deactivate

