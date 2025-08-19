#!/bin/bash

declare -a aud_map_type=("gaussian" "discrete" "neighbor")
declare -a rec_plasticity=("EIIE")
declare -a hvc_cond=("mature_hvc" "developing_hvc")

for tid in {0..2}
do
    for amt in "${aud_map_type[@]}"
    do
        for rp in "${rec_plasticity[@]}"
        do
            for hc in "${hvc_cond[@]}"
            do
                echo $amt $rp $hc
                python3 train_models.py $tid $amt $rp $hc
            done
        done
    done
done