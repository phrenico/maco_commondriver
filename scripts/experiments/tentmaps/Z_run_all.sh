#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 11 items
ZERO='[-----------] (0/11)'
ONE='[#----------] (1/11)'
TWO='[##---------] (2/11)'
THREE='[###--------] (3/11)'
FOUR='[####-------] (4/11)'
FIVE='[#####------] (5/11)'
SIX='[######-----] (6/11)'
SEVEN='[#######----] (7/11)'
EIGHT='[########---] (8/11)'
NINE='[#########--] (9/11)'
TEN='[##########-] (10/11)'
ELEVEN='[###########] (11/11)'

# define a function it has one arg that is a string and it is clearing the screen and echoing the string
function pbar {
    clear
    echo $1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"



# Run the files

# ICA
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_ica_res
conda deactivate

pbar $ONE
echo "ICA done"

# PCA
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_pca_res
conda deactivate

pbar $TWO
echo "ICA done"
echo "PCA done"

# KPCA
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_kpca_res
conda deactivate

pbar $THREE
echo "ICA done"
echo "PCA done"
echo "KPCA done"

# CCA
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_cca_res
conda deactivate

pbar $FOUR
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"

# DCA
conda activate dca
python -m scripts.experiments.tentmaps.gen_dca_res
conda deactivate

pbar $FIVE
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"

# DCCA
conda activate dcca_env
python3 -m scripts.experiments.tentmaps.gen_dcca_res
conda deactivate

pbar $SIX
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"


# Sh-Rec
conda activate shrec
python -m scripts.experiments.tentmaps.gen_shrec_res
conda deactivate

pbar $SEVEN
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"


# Random Control
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_random_res
conda deactivate

pbar $EIGHT
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"

# sfa
conda activate sfa
python3 -m scripts.experiments.tentmaps.gen_sfa_res
conda deactivate

pbar $NINE
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"

# MaCo
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_maco_res
conda deactivate

pbar $TEN
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"
echo "MaCo done"

# AniSOM
conda activate maco_rev1
python -m scripts.experiments.tentmaps.gen_anisom_res
conda deactivate

pbar $ELEVEN
echo "ICA done"
echo "PCA done"
echo "KPCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"
echo "MaCo done"
echo "AniSOM done"

conda activate maco_rev1
python -m scripts.experiments.tentmaps.Z_combine_final_res
conda deactivate






