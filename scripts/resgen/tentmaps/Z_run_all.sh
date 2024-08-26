#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 9 items
ZERO='[---------] (0/10)'
ONE='[#--------] (1/10)'
TWO='[##-------] (2/10)'
THREE='[###------] (3/10)'
FOUR='[####-----] (4/10)'
FIVE='[#####----] (5/10)'
SIX='[######---] (6/10)'
SEVEN='[#######--] (7/10)'
EIGHT='[########-] (8/10)'
NINE='[#########] (9/10)'
TEN='[##########] (10/10)'

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
## MaCo
#conda activate maco_rev1
#python gen_maco_res.py
#conda deactivate
#
#pbar $NINE
#
## AniSOM
#conda activate maco_rev1
#python gen_anisom_res.py
#conda deactivate
#
#pbar $TEN






