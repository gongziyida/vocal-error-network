`train_models.py` is a script that directly train all four types of models. One can run 

`python3 train_models.py TID AUD_MAP_TYPE ALT_REC_PLASTICITY HVC_COND` 

to save the trained models. It takes four parameters:
- `TID` specifies the ID for this task
- `AUD_MAP_TYPE`: one of `neighbor` (used in the main figures), `gaussian`, and `discrete`. For details see `src/train_funcs.read_realistic_input`
- `ALT_REC_PLASTICITY`: currently only is supported `EIIE`
- `HVC_COND`: one of `mature_hvc` (HVC bursting is regular throughout training; the case in the main figures) and 
`developing_hvc` (HVC bursting is gradually regular throughout training; a case shown in the supplementary information)
Change `RESULT_DIR = '../results/'` to your own saving directory. The results will be saved in `RESULT_DIR/trained_models_<AUD_MAP_TYPE>_map_<ALT_REC_PLASTICITY>_<HVC_COND>_<TID>.pkl`.

`run_training.sh` provides a bash script for running all of the conditions for 3 times (`TID` = 0, 1, 2). If you use this script to train the models, 
you can directly run `test_models.ipynb` to reproduce the plots in the paper. 
The other jupyter notebooks contain other examples and the code to generate some plots in the main and supplementary figures.
