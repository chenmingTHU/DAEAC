# DAEAC
This is an implementation of paper *Inter-patient ECG Arrhythmia Heartbeat Classification Based on Unsupervised Domain Adaptation*.

######

## Train & Test

In this implementation, we use *.yaml* files for configuration. Run the following codes for training and testing:


```shell
CUDA_VISIBLE_DEVICES=$1 python train.py --config=${CONFIG_FILE}

CUDA_VISIBLE_DEVICES=$1 python eval.py --config=${CONFIG_FILE}
```

## Data Preprocessing

We use three datasets in this work, including [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/), [INCARTDB](https://physionet.org/content/incartdb/1.0.0/) and [SVDB](https://physionet.org/content/svdb/1.0.0/). The users can download the original datasets in the above websits.

All original records are converted into *.mat* format with src/data/preprocessing.py, and the indices of heartbeats are saved as *.npz* files.
