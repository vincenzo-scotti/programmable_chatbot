import os
import sys
from shutil import copy2
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace

import yaml
import pickle
from typing import Optional, Dict

import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel

from programmable_chatbot.chatbot_api import Chatbot
from programmable_chatbot.data import PromptedOpenDomainDialogues
from programmable_chatbot.utils import nll, LinearAnnealingLR


# Global
random_seed: Optional[int] = None
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision: bool = True
checkpoint_gradient: bool = False
writer: SummaryWriter
# Chatbot
chatbot_configs: Dict
model_configs: Dict
model: PreTrainedModel
tokenizer_configs: Dict
tokenizer: PreTrainedTokenizer
# Data
corpus_configs: Dict
corpora: Dict[str, PromptedOpenDomainDialogues] = dict()
corpus_loaders: Dict[str, DataLoader] = dict()
# Optimisation
optimizer_configs: Dict
optimizer: Optimizer
scaler: Optional[GradScaler] = None
lr_scheduler_configs: Optional[Dict] = None
lr_scheduler: Optional[LinearAnnealingLR] = None
early_stopping_configs: Optional[Dict] = None
evaluation_configs: Dict
# Experiment dir path
current_experiment_dir_path: str
# Checkpoint path
model_checkpoint_path: str
best_model_checkpoint_path: str
# Results path
results_file_path: str


def init_environment(config_file_path: str):
    global random_seed, device, mixed_precision, checkpoint_gradient, writer
    global chatbot_configs, model_configs, tokenizer_configs
    global model, tokenizer
    global corpus_configs, corpora, corpus_loaders
    global optimizer_configs, lr_scheduler_configs, early_stopping_configs, evaluation_configs
    global optimizer, scaler, lr_scheduler
    global current_experiment_dir_path, model_checkpoint_path, best_model_checkpoint_path, results_file_path

    # Define helper function to create directory if not exits
    def mkd(path: str) -> str:
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file
    with open(config_file_path) as f:
        configs_dump_str: str = f.read()
        f.seek(0)
        configs: Dict = yaml.full_load(f)
    # Create directories
    # Main
    experiments_dir_path: str = mkd(configs['experiments_directory_path'])
    experiment_series_dir_path: str = mkd(os.path.join(experiments_dir_path, configs['experiment_series']))
    current_experiment_dir_path = mkd(os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time_experiment}"
    ))
    # Model
    model_dir_path: str = mkd(os.path.join(current_experiment_dir_path, 'model'))
    model_checkpoint_path = mkd(os.path.join(model_dir_path, 'latest_checkpoint'))
    best_model_checkpoint_path = mkd(os.path.join(model_dir_path, 'best_checkpoint'))
    # Logging
    # Tensorboard
    tb_dir_path = mkd(os.path.join(current_experiment_dir_path, 'tensorboard'))
    # Create file paths
    if configs.get('log_file', False):
        log_file_path = os.path.join(
            current_experiment_dir_path, f"{configs['experiment_id']}_{date_time_experiment}.log"
        )
    else:
        log_file_path = None
    configs_dump_path = os.path.join(current_experiment_dir_path, 'configs.yaml')
    # Init logging
    logging.basicConfig(filename=log_file_path, level=configs['log_level'])
    # Start Logging info
    logging.info(f"{configs['experiment_series']} training script started")
    logging.info(f"Current experiment directories created at '{current_experiment_dir_path}'")
    if log_file_path is not None:
        logging.info(f"Current experiment log created at '{log_file_path}'")
    # Results
    results_file_path = os.path.join(current_experiment_dir_path, 'results.pkl')
    # Set all random seeds
    random_seed = configs.get('random_seed', None)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    logging.info("Random seeds set")
    # Tensor-Board writer
    writer = SummaryWriter(tb_dir_path)
    logging.info(f"Tensor-Board writer created at '{tb_dir_path}'")
    # Dump configs
    copy2(config_file_path, configs_dump_path)
    writer.add_text('Configs', f"<pre>{configs_dump_str}</pre>")
    logging.info(f"Current experiment configuration dumped at '{configs_dump_path}'")
    # Set device
    device = torch.device(configs['device']) if 'device' in configs else device
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Check for gradient checkpointing
    checkpoint_gradient = configs.get('checkpoint_gradient', checkpoint_gradient)
    logging.info(f"Gradient checkpointing set to {checkpoint_gradient}")
    # Load remaining configs
    chatbot_configs = configs['chatbot']
    model_configs = configs['gpt2']['model']
    tokenizer_configs = configs['gpt2']['tokeniser']
    corpus_configs = configs['corpus']
    optimizer_configs = configs['optimizer']
    lr_scheduler_configs = configs.get('lr_scheduler')
    early_stopping_configs = configs.get('early_stopping')
    evaluation_configs = configs.get('evaluation', dict())
    # Run configuration steps
    configure_model()
    configure_data()
    configure_optimizer()

    logging.info("Initialisation completed")


def configure_model():
    global model, tokenizer
    # Create tokeniser instance
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_configs['pretrained'], **tokenizer_configs.get('kwargs', {})
    )
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = AutoModelForCausalLM.from_pretrained(
        model_configs['pretrained'], **model_configs.get('kwargs', {})
    )
    logging.info("Model instantiated")
    # Possibly enable gradient checkpointing
    if checkpoint_gradient:
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpoint enabled")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")

    return model, tokenizer


def configure_data():
    # Iterate over splits
    for split in ['train', 'validation', 'test']:
        # Create data set instance
        data_set: PromptedOpenDomainDialogues = PromptedOpenDomainDialogues(
            corpus_configs['corpora_dir_path'],
            tokenizer,
            split,
            corpus_configs['cache_dir_path'],
            augmentation=split == 'train',
            dropout=split == 'train',
            evaluation=split == 'test',
            device=device,
            **corpus_configs.get('kwargs', dict())
        )
        logging.info(f"{split.capitalize()} data set instantiated")
        # Add created data set and data loader to dict
        corpora[split] = data_set
        logging.info(f"{split.capitalize()} data set added to dictionary")
        if split != 'test':
            # Create data loader instance
            data_loader: DataLoader = DataLoader(
                data_set,
                batch_size=corpus_configs['data_loader'][split]['mini_batch_size'],
                num_workers=corpus_configs['data_loader'][split]['n_workers'],
                shuffle=split == 'train',
                collate_fn=data_set.collate
            )
            logging.info(f"{split.capitalize()} data loader instantiated")
            # Add created data loader to dict
            corpus_loaders[split] = data_loader
            logging.info(f"{split.capitalize()} data loader added to dictionary")
    logging.info("All data loaders instantiated")


def configure_optimizer():
    global optimizer, lr_scheduler, scaler
    # Create optimiser instance
    optimizer = torch.optim.AdamW(
        params=model.parameters(), **optimizer_configs['kwargs']
    )
    logging.info(f"Optimiser instantiated")
    # Create learning rate scheduler instance if required
    if lr_scheduler_configs is not None:
        # Get total number of training steps
        n_epochs = optimizer_configs['n_epochs']
        n_iterations = len(corpus_loaders['train'])
        n_accumulation = optimizer_configs.get('accumulation_steps', 1)
        steps = n_epochs * int(math.ceil(n_iterations / n_accumulation)) + 1
        # Update learning rate scheduler configs with missing info
        lr_scheduler_configs['steps'] = steps
        lr_scheduler = LinearAnnealingLR(optimizer, **lr_scheduler_configs)
        logging.info(f"Learning rate scheduler instantiated")

    # Create scaler if using mixed precision
    if mixed_precision:
        scaler = GradScaler()
        logging.info("Gradient scaler for mixed precision instantiated")

    return optimizer, lr_scheduler, scaler


def training_step(train_batch, loss_weight: float = 1.0, update: bool = True) -> torch.tensor:
    # Parameters
    max_gradient_norm = optimizer_configs.get('max_gradient_norm', 0)
    # Move tensors to device
    input_encodings, labels = train_batch
    input_encodings = input_encodings.to(device)
    labels = labels.to(device)
    # Process mini-batch
    with torch.autocast(device.type, enabled=mixed_precision):
        # Process current elements
        logits = model(**input_encodings, use_cache=checkpoint_gradient).logits
        # Compute loss
        loss = nll(logits, labels, reduction='mean')
        # Scale loss is required
        if loss_weight != 1:
            loss *= loss_weight
    logging.debug('Mini-batch processed')
    # Compute gradients
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    logging.debug('Gradient computed')
    # Detach loss
    loss = loss.detach()
    if update:
        # Clip gradient norm if required
        if max_gradient_norm > 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
        logging.debug('Gradient norm scaled')
        # Update weights
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        logging.debug('Weights updated')
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            logging.debug('Learning rate scheduler advanced')
        # Reset optimiser and model gradients
        for param in model.parameters():
            param.grad = None
        logging.debug('Gradient tape cleared')

    return loss


def validation_step(validation_batch) -> torch.tensor:
    # Move tensors to device
    input_encodings, labels = validation_batch
    input_encodings = input_encodings.to(device)
    labels = labels.to(device)
    # Process mini-batch
    with torch.autocast(device.type, enabled=mixed_precision):
        # Process current elements
        logits = model(**input_encodings).logits
        # Compute loss
        loss = nll(logits, labels, reduction='mean')
    logging.debug('Mini-batch processed')

    return loss


def fit():
    # Parameters
    train_corpus_loader = corpus_loaders['train']
    validation_corpus_loader = corpus_loaders['validation']
    n_epochs = optimizer_configs.get('n_epochs', 0)
    n_iterations = len(train_corpus_loader)
    n_accumulation = optimizer_configs.get('accumulation_steps', 1)
    validation_period = evaluation_configs.get('validation_period', n_iterations)
    logging_period = evaluation_configs.get('logging_period', n_iterations)
    best_validation_loss = float('inf')
    # Early stopping
    early_stopping_patience = early_stopping_configs.get('patience')
    early_stopping_counter = early_stopping_patience
    min_improvement = early_stopping_configs.get('min_improvement', 0.0)
    # Loss accumulator
    loss = []
    mini_batch_loss = None
    # Run initial validation step
    # evaluation_step()
    # Init step counter
    step_idx: int = 0
    # Train and validation process
    # Set model in training mode
    model.train()
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of training
    logging.info(f"Training started - Current date and time {start_time}")
    # Loop over epochs
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs} started")
        # Iterate over mini-batches
        for b_idx, mini_batch in enumerate(train_corpus_loader):
            # Parameters
            loss_weight = 1 / n_accumulation
            training_completed = (epoch == n_epochs - 1) and (b_idx == n_iterations - 1)
            update = (b_idx + 1) % n_accumulation == 0 or training_completed
            # Process current mini-batch
            if mini_batch_loss is None:
                mini_batch_loss = training_step(mini_batch, loss_weight=loss_weight, update=update)
            else:
                mini_batch_loss += training_step(mini_batch, loss_weight=loss_weight, update=update)
            logging.debug(f"Training mini-batch {b_idx + 1}/{n_iterations} processed")
            # Check whether the model was updated
            if update:
                # Append loss for logging
                loss.append((step_idx + 1, mini_batch_loss))
                # Reset loss
                mini_batch_loss = None
                # Update step counter
                step_idx += 1
                logging.debug(f"Training iteration step {step_idx} done")
                # Log loss if required
                if step_idx % logging_period == 0 or training_completed:
                    # Log info (iteration level) on Tensorboard
                    for iteration_idx, iteration_loss in loss:
                        writer.add_scalar(f'Loss/Training', iteration_loss.item(), iteration_idx)
                    # Clear accumulators
                    loss.clear()
                # Do validation step if required
                if step_idx % validation_period == 0 or training_completed:
                    # Disable gradient computation
                    with torch.no_grad():
                        # Set model in evaluation mode
                        model.eval()
                        # Log start of validation
                        logging.info(
                            f"Validation started"
                        )
                        # Iterate over validation mini-batches
                        validation_loss = torch.tensor([
                            validation_step(mini_batch) for mini_batch in validation_corpus_loader
                        ], device=device).mean().item()
                        # Log end of validation
                        logging.info(
                            f"Validation finished - step {step_idx}"
                        )
                        #
                        writer.add_scalar(f'Loss/Validation', validation_loss, step_idx)
                        # Set model back in training mode
                        model.train()
                    # Compare new current best with old best
                    if best_validation_loss - validation_loss < min_improvement:
                        if early_stopping_counter is not None:
                            early_stopping_counter -= 1 if epoch > 0 else 0
                    # Else if improved keep training
                    else:
                        # Update best loss
                        best_validation_loss = validation_loss
                        # Checkpoint best model
                        model.save_pretrained(best_model_checkpoint_path)
                        logging.info("Validation objective improved, best model saved")
                        # Reset counter
                        early_stopping_counter = early_stopping_patience
                    # If the counter is exhausted activate early stopping
                    if early_stopping_counter is not None and not training_completed and early_stopping_counter == 0:
                        # Checkpoint trained model
                        model.save_pretrained(model_checkpoint_path)
                        logging.info("Models saved using utilities")
                        # Log early stopping
                        logging.info(f"Early stopping activated")
                        # Restore best validation model weights
                        model.load_state_dict(torch.load(os.path.join(best_model_checkpoint_path, 'pytorch_model.bin')))
                        model.to(device)
                        logging.info("Best validation model weights restored")
                        # Close training
                        # Get current date and time
                        end_time: datetime = datetime.now()
                        # Log end of training
                        logging.info(f"Training finished - Current date and time {end_time}")
                        logging.info(f"Elapsed time {end_time - start_time}")

                        return
        # Checkpoint trained model
        model.save_pretrained(model_checkpoint_path)
        logging.info("Models saved using utilities")
        # Log end of epoch
        logging.info(f"Epoch {epoch + 1}/{n_epochs} finished")
    # Log end of fit
    logging.info(f"Training finished")
    # Restore best validation model weights
    model.load_state_dict(torch.load(os.path.join(best_model_checkpoint_path, 'pytorch_model.bin')))
    model.to(device)
    logging.info("Best validation model weights restored")
    # Close training
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of training
    logging.info(f"Training finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")

    return


def evaluate():
    # Parameters
    test_data = corpora['test'].data
    # Accumulator
    results = dict()
    # Test process
    # Set model in evaluation mode
    model.eval()
    # Create chatbot instance
    chatbot = Chatbot(
        model, tokenizer=tokenizer, device=device, mixed_precision=mixed_precision, **chatbot_configs['kwargs']
    )
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of testing
    logging.info(f"Testing started - Current date and time {start_time}")
    # Iterate over data sets
    for data_set in test_data:
        logging.debug(f"Testing started data set {data_set}")
        # Data set specific dict for results
        results[data_set] = dict()
        # Run test on generative model (if available)
        if test_data[data_set].get('generator') is not None:
            results[data_set]['generator'] = dict()
            for approach in ['plain', 'conditioned']:
                results[data_set]['generator'][approach] = chatbot.eval_generator(
                    test_data[data_set]['generator'][approach]
                ) if test_data[data_set]['generator'].get(approach) is not None else None
            logging.debug(f"Generative model test complete")
        else:
            results[data_set]['generator'] = None
        # Run test on discriminative model (if available)
        if test_data[data_set].get('discriminator') is not None:
            results[data_set]['discriminator'] = dict()
            for lbl_type in ['global', 'local']:
                if test_data[data_set]['discriminator'].get(lbl_type) is not None:
                    results[data_set]['discriminator'][lbl_type] = {
                        lbl: {
                            approach: chatbot.eval_discriminator(
                                test_data[data_set]['discriminator'][lbl_type][lbl][approach],
                                lbl_type,
                                approach
                            )
                            for approach in test_data[data_set]['discriminator'][lbl_type][lbl]
                        }
                        for lbl in test_data[data_set]['discriminator'][lbl_type]
                    }
                else:
                    results[data_set]['discriminator'][lbl_type] = None
            logging.debug(f"Discriminative model test complete")
        else:
            results[data_set]['discriminator'] = None
        # Run test on explanations model (if available)
        if test_data[data_set].get('explanations') is not None:
            results[data_set]['explanations'] = dict()
            for lbl_type in ['global', 'local']:
                if test_data[data_set]['explanations'].get(lbl_type) is not None:
                    results[data_set]['explanations'][lbl_type] = {
                        lbl: chatbot.eval_explanations(test_data[data_set]['explanations'][lbl_type][lbl])
                        for lbl in test_data[data_set]['explanations'][lbl_type]
                    }
                else:
                    results[data_set]['explanations'][lbl_type] = None
            logging.debug(f"Explanations model test complete")
        else:
            results[data_set]['explanations'] = None
        logging.debug(f"Testing completed data set {data_set}")
    # Serialise results dict into binary file
    with open(results_file_path, 'bw') as f:
        pickle.dump(results, f)
    logging.debug("Results dictionary saved")
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of training
    logging.info(f"Training finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")

    return


def main(args: Namespace):
    # Prepare environment
    init_environment(args.config_file_path)
    # Run training and validation
    # fit()
    # Run tests
    evaluate()

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the experiment."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
