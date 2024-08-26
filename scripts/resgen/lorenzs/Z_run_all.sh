#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 9 items
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



# Run the files

## ICA
#conda activate maco_rev1
#python gen_ica_res.py
#conda deactivate
#
#pbar $ONE
#
## PCA
#conda activate maco_rev1
#python gen_pca_res.py
#conda deactivate
#
#pbar $TWO
#
## CCA
#conda activate maco_rev1
#python gen_cca_res.py
#conda deactivate
#
#pbar $THREE
#
## DCA
#conda activate dca
#python gen_dca_res.py
#conda deactivate
#
#pbar $FOUR
#
## DCCA
#conda activate dcca_env
#python gen_dcca_res.py
#conda deactivate
#
#pbar $FIVE
#
## Sh-Rec
#conda activate shrec
#python gen_shrec_res.py
#conda deactivate
#
#pbar $SIX
#
#
## Random Control
#conda activate maco_rev1
#python gen_random_res.py
#conda deactivate
#
#pbar $SEVEN
#
## sfa
#conda activate sfa
#python gen_sfa_res.py
#conda deactivate
#
#pbar $EIGHT
#
# MaCo
conda activate maco_rev1
python gen_maco_res.py
conda deactivate

#pbar $NINE






