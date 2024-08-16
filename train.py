# Your imports and other code here
import os
import torch
import yaml
import multiprocessing as mp
import argparse
from dataset import get_dataloader
from fairseq.models import BaseFairseqModel
from fairseq.models.wav2vec import TransformerEncoder
from dinosr import DinoSR, model_creator
from dinosr_config import DinosrAudioConfig
from torch import optim
from model_persistant_state import ModelPersistantState
from mamba import DeepMamba
from typing import List
from matplotlib import pyplot as plt
import sys

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetricMemory:
    def __init__(
        self,
        metric_name: str,
        memory_window: int,
    ):
        self.metric = None
        if len(metric_name) > 40:
            raise ValueError("The metric name is too long, must be less than 40 characters")
        # make self.metric name 40 chcaracters long, starting with the metric name
        self.metric_name = metric_name + ' ' * (30 - len(metric_name))
        self.memory_window = memory_window

    def update(
        self,
        metric_step
    ):
        if self.metric is None:
            self.metric = metric_step
        else:
            tau = 1 / self.memory_window
            self.metric = metric_step * tau + self.metric * (1 - tau)
        return self.metric

    def _get_str(
        self,
        print_precentage: bool,
    ): # print it in a table format
        metric_str = f"{(self.metric*100):.2f}%" if print_precentage else f"{self.metric:.4f}"
        if len(metric_str) < 10:
            metric_str = metric_str + ' ' * (10 - len(metric_str)) 
        else:
            metric_str = metric_str[:10]
        return f"|    {self.metric_name} |    {metric_str}|"

    def print(
        self,
        print_precentage: bool = False
    ):
        print(self._get_str(print_precentage))
        

    def get_print_length(self):
        return len(self._get_str(False)) 

    def get_metric(self):
        return self.metric


# using argparse make it run in two modes, load or create. If load is specified, then a model name must be specified
# if create is specified, then a config file must be specified along with a model name.

argparser = argparse.ArgumentParser()
# make the mode excepting only two values, load or create
argparser.add_argument('mode', type=str, help='Mode to run the script in, either load or create', choices=['load', 'create'])
# argparser.add_argument('mode', type=str, help='Mode to run the script in, either load or create')
argparser.add_argument('model_name', type=str, help='Name of the model to load')
argparser.add_argument('--config_file', type=str, help='Path to the config file to create the model, must be specified when creating a model')
args = argparser.parse_args()

# Guard the main code
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model_name
    model_path = f'./models/{model_name}'
    model_exists = os.path.exists(model_path)
    model_persistant_state = ModelPersistantState(model_path)
    if args.mode == 'load':
        try:
            dino_model = model_persistant_state.load_model()
            cfg = dino_model.cfg
            print("Loaded model successfully")
        except:
            print(f"Model {model_name} does not exist")
            sys.exit(1)
    else:
        if args.config_file is None:
            raise ValueError("Config file must be specified when creating a model")
        if model_exists:
            print(f"Model {model_name} already exists, are you sure you want to overwrite it? (y/N)")
            response = input()
            if response.lower() != 'y':
                sys.exit(1)
            else:
                os.system(f"rm -rf {model_path}/*")
        if not os.path.exists(args.config_file):
            raise ValueError("Config file does not exist")
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        cfg = DinosrAudioConfig(**config)
        dino_model = DinoSR(cfg, model_creator).to(device)
        print("Created model successfully")

    dino_model.to(device)

    num_epochs = 50
    batch_size = 320
    mini_batch_size = 16
    learning_rate = 0.0005

    # Define the learning rate schedule
    def lr_lambda(initial_step, step):
        warmup_steps = 12000
        hold_steps = 188000
        decay_steps = 200000
        initial_lr = 0.0005
        final_lr = 0.00005

        modified_step = step + initial_step

        if modified_step < warmup_steps:
            return modified_step / warmup_steps
        elif modified_step < warmup_steps + hold_steps:
            return 1.0
        else:
            decay_factor = (modified_step - (warmup_steps + hold_steps)) / decay_steps
            return initial_lr * ((final_lr / initial_lr) ** decay_factor) / initial_lr

    optimizer = optim.Adam(dino_model.parameters(), lr=0.0005)

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(model_persistant_state.get_current_step(), step))

    trainset = get_dataloader(mini_batch_size)

    total_step = len(trainset)
    n = batch_size // mini_batch_size  # Update parameters every n batches

    batch_step = model_persistant_state.get_current_step()

    long_loss = MetricMemory('Long Loss', 100)
    long_accuracy = MetricMemory('Long Accuracy', 100)
    long_kl_loss = MetricMemory('Long KL Loss', 100)
    long_ce_loss = MetricMemory('Long CE Loss', 100)
    short_loss = MetricMemory('Short Loss', 10)
    short_accuracy = MetricMemory('Short Accuracy', 10)
    short_kl_loss = MetricMemory('Short KL Loss', 10)
    short_ce_loss = MetricMemory('Short CE Loss', 10)

    active_codewords = MetricMemory('Active Codewords', 1)

    # make plots dir
    os.makedirs(f'{model_path}/plots', exist_ok=True)
    for epoch in range(num_epochs):
        mb_loss = 0
        mb_accuracy = 0
        mb_kl_loss = 0
        mb_ce_loss = 0
        prob_bins = torch.zeros(cfg.codebook_size).to(device)
        prob_bins_binary = torch.zeros(cfg.codebook_size).to(device).to(torch.bool)
        for i, (waveforms, lengths) in enumerate(trainset):
            # empty cache to avoid memory leak
            torch.cuda.empty_cache()

            # Forward pass
            results = dino_model(waveforms, lengths)
            loss = results['loss'] / n
            accuracy = results['accuracy']
            prob_bins += results['prob_bins']
            prob_bins_binary |= results['prob_bins_binary']
            target = results['targets'][-1]
            onehot_target = torch.nn.functional.one_hot(target, num_classes=cfg.codebook_size)


            
            # Accumulate loss and accuracy
            mb_loss += loss.item()
            mb_accuracy += accuracy.item()
            mb_kl_loss += results['kl_divergence_loss'].item() / n
            mb_ce_loss += results['cross_entropy_loss'].item() / n

            # Accumulate codewords for update
            for layer_idx in results['codebook_update']:
                x = results['codebook_update'][layer_idx]['flattened_teacher_layer_results']
                closest_codewords = results['codebook_update'][layer_idx]['closest_codewords']
                dino_model.codebook.accumulate_codewords_for_update(
                    layer_idx, 
                    x, 
                    closest_codewords, 
                )
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients and update parameters
            if (i + 1) % n == 0 or (i + 1) == total_step:
                # Increment the batch step
                batch_step += 1
                
                optimizer.step()
                dino_model.update_teacher_params(batch_step)
                optimizer.zero_grad()
                
                # Step the scheduler
                scheduler.step()

                # Update the codebooks
                dino_model.codebook.update_codewords()

                # update the metrics
                long_loss.update(mb_loss / n)
                long_accuracy.update(mb_accuracy / n)
                long_kl_loss.update(mb_kl_loss / n)
                long_ce_loss.update(mb_ce_loss / n)
                short_loss.update(mb_loss / n)
                short_accuracy.update(mb_accuracy / n)
                short_kl_loss.update(mb_kl_loss / n)
                short_ce_loss.update(mb_ce_loss / n)

                active_codewords.update(
                    torch.sum(prob_bins_binary.to(torch.float32)).to("cpu").detach().numpy() / cfg.codebook_size
                )


                # zero the metrics for next step
                mb_loss = 0
                mb_accuracy = 0
                mb_kl_loss = 0
                mb_ce_loss = 0

                # Save the model and training history
                model_persistant_state.save_model(
                    step=batch_step,
                    model=dino_model,
                    performance={
                        'loss': long_loss.get_metric(),
                        'accuracy': long_accuracy.get_metric(),
                    }
                )    

                # Print the metrics
                print("=" * long_loss.get_print_length())
                title_string = f"Epoch [{epoch + 1}/{num_epochs}], Batch step [{batch_step}]"
                print(f"| {title_string}" + " " * (long_loss.get_print_length() - len(title_string) - 3) + "|")
                long_loss.print()
                long_accuracy.print(print_precentage=True)
                long_kl_loss.print()
                long_ce_loss.print()
                print("-" * long_loss.get_print_length())
                short_loss.print()
                short_accuracy.print(print_precentage=True)
                short_kl_loss.print()
                short_ce_loss.print()
                print("-" * long_loss.get_print_length())
                active_codewords.print(print_precentage=True)

                # print the targets
                prob_bins = prob_bins / cfg.average_top_k_layers
                prob_bins = prob_bins.to("cpu").detach().numpy()
                number_of_active_bins = torch.sum(prob_bins_binary.to(torch.float32)).item()
                prob_bins_binary = prob_bins_binary.to(torch.float32).to("cpu").detach().numpy()
                # plot as a histogram to the console
                plt.bar(range(cfg.codebook_size), prob_bins)
                # save plot to plt.png
                plt.savefig(f'{model_path}/plots/plt_{batch_step}.png')
                # clear the plot
                plt.clf()

                plt.bar(range(cfg.codebook_size), prob_bins_binary)
                # save plot to plt.png
                plt.savefig(f'{model_path}/plots/plt_binary_{batch_step}.png')
                plt.clf()

                # delete old plot
                old_batch_step = batch_step - 10
                if old_batch_step > 0:
                    os.system(f"rm -rf {model_path}/plots/plt_{old_batch_step}.png")
                    os.system(f"rm -rf {model_path}/plots/plt_binary_{old_batch_step}.png")

                # print("The number of active bins is: ", number_of_active_bins)
                
                prob_bins = torch.zeros(cfg.codebook_size).to(device)
                prob_bins_binary = torch.zeros(cfg.codebook_size).to(device).to(torch.bool)
                
