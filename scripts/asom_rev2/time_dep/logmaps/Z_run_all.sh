#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 9 items
ZERO='[----------] (0/10)'
ONE='[#---------] (1/10)'
TWO='[##--------] (2/10)'
THREE='[###-------] (3/10)'
FOUR='[####------] (4/10)'
FIVE='[#####-----] (5/10)'
SIX='[######----] (6/10)'
SEVEN='[#######---] (7/10)'
EIGHT='[########--] (8/10)'
NINE='[#########-] (9/10)'
TEN='[###########] (10/10)'

# define a function it has one arg that is a string and it is clearing the screen and echoing the string
function pbar {
    clear
    echo $1
}




# Run the files

# ICA
conda activate maco_rev1
python gen_ica_res.py
conda deactivate

pbar $ONE
echo "ICA done"

# PCA
conda activate maco_rev1
python gen_pca_res.py
conda deactivate

pbar $TWO
echo "ICA done"
echo "PCA done"

# CCA
conda activate maco_rev1
python gen_cca_res.py
conda deactivate

pbar $THREE
echo "ICA done"
echo "PCA done"
echo "CCA done"

# DCA
conda activate dca
python gen_dca_res.py
conda deactivate

pbar $FOUR
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"

# DCCA
conda activate dcca_env
python3 gen_dcca_res.py
conda deactivate

pbar $FIVE
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"


# Sh-Rec
conda activate shrec
python gen_shrec_res.py
conda deactivate

pbar $SIX
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"


# Random Control
conda activate maco_rev1
python gen_random_res.py
conda deactivate

pbar $SEVEN
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"

# sfa
conda activate sfa
python3 gen_sfa_res.py
conda deactivate

pbar $EIGHT
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"

# MaCo
conda activate maco_rev1
python gen_maco_res.py
conda deactivate

pbar $NINE
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"
echo "MaCo done"

# AniSOM
conda activate maco_rev1
python gen_anisom_res.py
conda deactivate

pbar $TEN
echo "ICA done"
echo "PCA done"
echo "CCA done"
echo "DCA done"
echo "DCCA done"
echo "Sh-Rec done"
echo "Random done"
echo "SFA done"
echo "MaCo done"
echo "AniSOM done"

conda activate maco_rev1
python Z_combine_final_res.py
conda deactivate





