# Phi-3-Finetuning-on-CoEdit-Dataset

This repo contains a Python script to finetune the Phi 3 model on the CoEdit Dataset. The goal is to make the Phi 3 model work well on text editing in the English language by fine-tuning it on the Coedit Dataset.

The script is based on the [notebook](https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing) of Unsloth AI and was run on a Google Cloud Machine with L4 GPU. The checkpoint was saved [here](https://huggingface.co/letheviet/coedit-Phi-3-mini-4k-instruct-fulltrain) which was trained on full the CoEdit dataset on 2 epochs.