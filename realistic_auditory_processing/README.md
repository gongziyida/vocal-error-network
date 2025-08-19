# Generate sparse embeddings of spectrograms of real adult bird song syllables

*If your goal is just to simulate our network models using realistic data, you can download the data* [here](https://app.box.com/s/9uu0zwzoqw5qd4ghijg220juxy912mye). If you want to learn more about the process to generate sparse embeddings of spectrograms of real adult bird song syllables, you can read below.

We used a combination of adult songs and audios recorded around young birds (sound due to behaviors and calls). Labelled adult songs are available at this [dataset](https://doi.org/10.18738/T8/SAWMUN) by Therese Koch and was preprocessed by `segment_song.py` to segment syllables and generate spectrograms. 

For audios recorded around young birds, we used [AVA](https://autoencoded-vocal-analysis.readthedocs.io/en/latest/index.html) to turn audios into spectrograms. The AVA processing takes three steps:
1. Tune the parameters for segmentation from raw audios (`ava_parameter_tuning.py`).
2. Perform sound segmentation and write segmentation decisions (`ava_segmentation.py`).
3. Compute the spectrograms for the segmented audios (`ava_prepare_spectrograms.py`). 
For more details, please refer to AVA's documentation.

After computing these spectrograms, we trained a sparse coding model to generate the sparse embeddings of songs in `train_sparse_coding.ipynb`. This Jupyter Notebook file contains the detailed steps to train and use the sparse coding model, and finally output `learned_song_responses.npz` used by simulations in this paper. 

If you only want to train the sparse coding model on adult bird songs, the AVA procedures can be skipped and a lighter version is in `train_sparse_coding_light.ipynb` (for demonstration only, not used in the paper).