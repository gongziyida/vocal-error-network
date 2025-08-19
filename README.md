# Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning

This repository contains the code for simulating the models and generating the core figures in the paper **Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning** [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.18.665446).

The mathematical details are described in the paper. `src` contains the source code for the networks, the reinforcement learning environment, the sparse coding model, and other helper functions. 

For the basic training and simulations of the models, check out examples in `simulations`, which also contains the code to generate the panels such as neuronal firing rates, and joint distributions of mean rates between correct and perturbed singing cases in the paper. Notice that the networks themselves are implemented to take any time series input that matches the dimensions, but the code in `simulations` uses sparse embeddings of real spectrograms of adult songs saved `realistic_auditory_processing/learned_song_responses.npz`. To generate this embedding, follow the steps in `realistic_auditory_processing`. Finally, `further_analysis` contains the code to run and visualize the analysis of the learned recurrent weights, as well as the reinforcement learning experiments. 
