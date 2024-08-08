# Your imports and other code here
import os
import torch
import yaml
import multiprocessing as mp

from dataset import get_dataloader
from fairseq.models import BaseFairseqModel
from fairseq.models.wav2vec import TransformerEncoder
from dinosr import DinoSR
from dinosr_config import DinosrAudioConfig
from torch import optim
from model_persistant_state import ModelPersistantState
from mamba import DeepMamba

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)


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



# Guard the main code
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = DinosrAudioConfig(
        average_top_k_layers=6,
        encoder_layers=8,
    )

    def model_creator(cfg, model_type="transformer"):
        if model_type == "transformer":
            return TransformerEncoder(cfg)
        else:
            return DeepMamba(cfg)

    dino_model = DinoSR(cfg, lambda cfg: model_creator(cfg, "transformer")).to(device)
    
    model_persistant_state = ModelPersistantState('./models/transformer')
    try:
        model_persistant_state.load_model(dino_model)
        print("Loaded model successfully")
    except:
        print("No model to load")

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
    long_prob_mean = MetricMemory('Long Prob Mean', 100)
    short_loss = MetricMemory('Short Loss', 10)
    short_accuracy = MetricMemory('Short Accuracy', 10)
    short_prob_mean = MetricMemory('Short Prob Mean', 10)
    for epoch in range(num_epochs):
        mb_loss = 0
        mb_accuracy = 0
        mb_prob_mean = 0
        for i, (waveforms, lengths) in enumerate(trainset):
            # empty cache to avoid memory leak
            torch.cuda.empty_cache()

            # Forward pass
            results = dino_model(waveforms, lengths)
            loss = results['loss'] / n
            accuracy = results['accuracy']
            prob_mean = results['prob_mean']
            
            # Accumulate loss and accuracy
            mb_loss += loss.item()
            mb_accuracy += accuracy.item()
            mb_prob_mean += prob_mean.item()
            
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

                # update the metrics
                long_loss.update(mb_loss / n)
                long_accuracy.update(mb_accuracy / n)
                long_prob_mean.update(mb_prob_mean / n)
                short_loss.update(mb_loss / n)
                short_accuracy.update(mb_accuracy / n)
                short_prob_mean.update(mb_prob_mean / n)

                # zero the metrics for next step
                mb_loss = 0
                mb_accuracy = 0
                mb_prob_mean = 0

                # Save the model and training history
                model_persistant_state.save_model(
                    step=batch_step,
                    model=dino_model,
                    performance={
                        'loss': long_loss.get_metric(),
                        'accuracy': long_accuracy.get_metric(),
                        'prob_mean': long_prob_mean.get_metric(),
                    }
                )    

                # Print the metrics
                print("=" * long_loss.get_print_length())
                title_string = f"Epoch [{epoch + 1}/{num_epochs}], Batch step [{batch_step}]"
                print(f"| {title_string}" + " " * (long_loss.get_print_length() - len(title_string) - 3) + "|")
                long_loss.print()
                long_accuracy.print(print_precentage=True)
                long_prob_mean.print(print_precentage=True)
                print("-" * long_loss.get_print_length())
                short_loss.print()
                short_accuracy.print(print_precentage=True)
                short_prob_mean.print(print_precentage=True)        
