#!/bin/sh

# This is a shell script that will run a bunch of trials with the following
# Seed and xavier scale values and save the output to the trials directory.
# You can use the command (jupyter nbconvert --to script CNN.ipynb) to conver
# The notebook to a python file which can be run in this script

# IMPORTANT: Before running the script change the values of the seed and 
# Xavier scaling paramter in CNN.py to be argv[1] and argv[2] respectively 

seed_vals="6 7 8"
xavier_scale_vals=" 0.1 0.5 1 5 15"

for seed_val in $seed_vals; do
    for xavier_scale_val in $xavier_scale_vals; do
        #conda init
        #conda activate ../envs/eds4ai_env/
        python CNN.py $seed_val $xavier_scale_val > junk.txt
    done
done
