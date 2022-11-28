# Programmable Chatbot

Codebase for the paper "[Building Programmable Open-Domain Conversational Agents for Chit-Chat and Mental Healthcare]()". 
This repository contains the implementation dialogue language model described in the paper.

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments;
    - checkpoints generated during the experiments;
    - experiment configuration dumps;
    - experiment logs.
- `notebooks/` contains the directories to host:  
    - data exploration notebooks.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
    - directory to host the pre-trained models, and the references to download them.
- `src/` contains modules and scripts to: 
    - run training and evaluation steps;
    - interact with the trained models;
    - data API
    - chatbot API

For further details, refer to the `README.md` within each directory.

## Environment

To install all the required packages within an anaconda environment, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n progchat python=3.10 cudatoolkit=11.6
# Activate anaconda environment
conda activate progchat
# Install packages
conda install pytorch pytorch-cuda=11.6 -c pytorch
conda install -c conda-forge transformers tensorboard scikit-learn matplotlib seaborn jupyterlab
```

To add the source code directory to the Python path, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/programmable_chatbot/src
```

## Training

### Run

There is a script to train or fine-tune the model, it expects to have `./src` in the Python path and all data sets to be downloaded and placed in the `./resources/data/raw/` directory.

To train or fine-tune the model run:
```bash
python ./src/bin/train.py --config_file_path ./resources/configs/path/to/training/config.yaml
```

To train or fine-tune the model in background run:

```bash
nohup python ./src/bin/train.py --config_file_path ./resources/configs/path/to/training/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

### Monitor

To connect to a remote server and monitor the training process via [Tensorboard](https://www.tensorflow.org/tensorboard) connect via ssh to your machine using a tunnel

```bash
ssh  -L 16006:127.0.0.1:6006 user@adderess
```

Start the Tensorboard server on the remote machine

```bash
tensorboard --logdir ./expertiments/path/to/tensorboard/
```

Finally connect to http://127.0.0.1:16006 on your local machine

## Evaluation

There is a script to run the final evaluation of the model, the requirements to run it are the same of the training script.

To run the evaluation in background execute:

```bash
nohup python ./src/bin/evaluate.py --config_file_path ./resources/configs/path/to/evaluation/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

## Chatting

There is a script available to chat directly with any of the models, it can be run using the following command:

```bash
python ./src/bin/interact.py --config_file_path ./resources/configs/path/to/inference/config.yaml
```

Alternatively there is the `chatbot_api` sub-module designed for the re-use of the agent outside the repository.
The API uses the base [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) class from the [Transformers](https://huggingface.co/docs/transformers/index) library.

Here follows a usage example

```python
...
```

## References

If you are willing to use our code or our models, please cite our work through the following BibTeX entry:

```bibtex
...
```