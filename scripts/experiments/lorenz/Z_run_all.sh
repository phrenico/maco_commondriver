#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 9 items
ZERO='[---------] (0/9)'
ONE='[#--------] (1/9)'
TWO='[##-------] (2/9)'
THREE='[###------] (3/9)'
FOUR='[####-----] (4/9)'
FIVE='[#####----] (5/9)'
SIX='[######---] (6/9)'
SEVEN='[#######--] (7/9)'
EIGHT='[########-] (8/9)\n'
NINE='[#########] (9/9)\n'

# define a function it has one arg that is a string and it is clearing the screen and echoing the string
function pbar {
    clear
    echo $1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"





# Run the files
pbar $ZERO

# ICA
conda activate maco_rev1
python -m scripts.experiments.lorenz.gen_ica_res
conda deactivate

pbar $ONE
echo "ICA - done."
echo "PCA - starting..."
echo "CCA - pending.."
echo "DCA - pending.."
echo "DCCA - pending.."
echo "Sh-Rec - pending.."
echo "Random - pending.."
echo "sfa - pending.."
echo "MaCo - pending.."


# PCA
conda activate maco_rev1
python -m scripts.experiments.lorenz.gen_pca_res
conda deactivate

pbar $TWO
echo "ICA - done."
echo "PCA - done."
echo "CCA - starting..."
echo "DCA - pending.."
echo "DCCA - pending.."
echo "Sh-Rec - pending.."
echo "Random - pending.."
echo "sfa - pending.."
echo "MaCo - pending.."

# CCA
conda activate maco_rev1
python -m scripts.experiments.lorenz.gen_cca_res
conda deactivate

pbar $THREE
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - starting..."
echo "DCCA - pending.."
echo "Sh-Rec - pending.."
echo "Random - pending.."
echo "sfa - pending.."
echo "MaCo - pending.."

# DCA
conda activate dca
python -m scripts.experiments.lorenz.gen_dca_res
conda deactivate

pbar $FOUR
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - starting..."
echo "Sh-Rec - pending.."
echo "Random - pending.."
echo "sfa - pending.."
echo "MaCo - pending.."


# DCCA
conda activate dcca_env
python3 -m scripts.experiments.lorenz.gen_dcca_res
conda deactivate

pbar $FIVE
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - done."
echo "Sh-Rec - starting..."
echo "Random - pending.."
echo "sfa - pending.."
echo "MaCo - pending.."

# Sh-Rec
conda activate shrec
python -m scripts.experiments.lorenz.gen_shrec_res
conda deactivate

pbar $SIX
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - done."
echo "Sh-Rec - done."
echo "Random - starting..."
echo "sfa - pending.."
echo "MaCo - pending.."


# Random Control
conda activate maco_rev1
python -m scripts.experiments.lorenz.gen_random_res
conda deactivate

pbar $SEVEN
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - done."
echo "Sh-Rec - done."
echo "Random - done."
echo "sfa - starting..."
echo "MaCo - pending.."

# sfa
conda activate sfa
python3 -m scripts.experiments.lorenz.gen_sfa_res
conda deactivate

pbar $EIGHT
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - done."
echo "Sh-Rec - done."
echo "Random - done."
echo "sfa - done."
echo "MaCo - starting..."


# MaCo
conda activate maco_rev1
python -m scripts.experiments.lorenz.gen_maco_res
conda deactivate

pbar $NINE
echo "ICA - done."
echo "PCA - done."
echo "CCA - done."
echo "DCA - done."
echo "DCCA - done."
echo "Sh-Rec - done."
echo "Random - done."
echo "sfa - done."
echo "MaCo - done."

conda activate maco_rev1
python -m scripts.experiments.lorenz.Z_combine_final_res
conda deactivate





