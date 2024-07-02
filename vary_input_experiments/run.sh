#!/bin/bash
for i in {1..5}
do
    python3 vary_input_dim.py $i
    python3 vary_percent_perturbed.py $i
done
