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

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Guard the main code
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = DinosrAudioConfig(
        average_top_k_layers=4,
        encoder_layers=4,
    )

    def model_creator(cfg):
        return TransformerEncoder(cfg)

    dino_model = DinoSR(cfg, model_creator).to(device)
    
    model_persistant_state = ModelPersistantState('./models/ext_transformer')
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

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        local_loss = 0.0
        local_accuracy = 0.0
        for i, (waveforms, lengths) in enumerate(trainset):
            step = epoch * total_step + i  # Calculate the current step

            # Forward pass
            results = dino_model(waveforms, lengths)
            loss = results['loss'] / n
            accuracy = results['accuracy']
            
            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            local_loss += loss.item()
            local_accuracy += accuracy.item()
            
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

                # Save the model and training history
                model_persistant_state.save_model(
                    step=batch_step,
                    model=dino_model,
                    performance={
                        'loss': epoch_loss / (i + 1),
                        'accuracy': epoch_accuracy / (i + 1)
                    }
                )            

                print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {epoch_loss / (i + 1):.4f}, Accuracy: {100 * epoch_accuracy / (i + 1):.2f}% local accuracy is {100 * local_accuracy / n:.2f}%, and local loss is: {local_loss / n:.4f}', end='', flush=True)
                local_accuracy = 0.0
                local_loss = 0.0

        # Calculate and print cumulative loss and accuracy for the epoch
        avg_loss = epoch_loss / total_step
        avg_accuracy = epoch_accuracy / total_step
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Summary: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {100 * avg_accuracy:.2f}%')
