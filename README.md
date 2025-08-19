# Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning

This repository contains the code for simulating the models and generating the core figures in the paper **Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning** [(bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.07.18.665446).

The mathematical details are described in the paper. `src` contains the source code for the networks, the reinforcement learning environment, the sparse coding model, and other helper functions. 

For the basic training and simulations of the models, check out examples in `simulations`, which also contains the code to generate the panels such as neuronal firing rates, and joint distributions of mean rates between correct and perturbed singing cases in the paper. Notice that the networks themselves are implemented to take any time series input that matches the dimensions, but the simulations to reproduce the figures in the paper uses sparse embeddings of real spectrograms of adult songs. To generate this embedding, follow the steps in `realistic_auditory_processing` or directly [download](https://app.box.com/s/9uu0zwzoqw5qd4ghijg220juxy912mye) the generated embeddings. 

`further_analysis` contains the code to run and visualize the analysis of the learned recurrent weights, as well as the reinforcement learning experiments. `robustness_tests` contains the code to plot the cancellation quality over different parameters (Fig. 3C).