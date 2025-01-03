import os
import yaml
from dinosr_config import DinosrAudioConfig
from dinosr import DinoSR, model_creator
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define functions and classes here
class ModelPersistantState:
    def __init__(self, model_dir):
        self.acc_model_path = model_dir + f"/acc_model.pt"
        self.loss_model_path = model_dir + f"/loss_model.pt"
        self.performance_path = model_dir + f"/performance.yaml"
        self.cfg_path = model_dir + f"/config.yaml"
        self.current_step = 0
        self.performance = {
            "best_accuracy": 0,
            "best_loss": 10000000000,
            "step": 0,
        }
        self.cfg_saved: DinosrAudioConfig = False

        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, step, model, performance):
        if performance['loss'] < self.performance['best_loss']:
            self.performance['best_loss'] = performance['loss']
            model.save(self.loss_model_path)
        if performance['accuracy'] > self.performance['best_accuracy']:
            self.performance['best_accuracy'] = performance['accuracy']
            model.save(self.acc_model_path)
        self.performance['step'] = step
        self.performance[f'accuracy_step={step}'] = performance['accuracy']
        self.performance[f'loss_step={step}'] = performance['loss']

        # Correctly open the file in write mode and dump the YAML
        with open(self.performance_path, 'w') as file:
            yaml.dump(self.performance, file)

        if not self.cfg_saved:
            self.cfg_saved = True
            model.cfg.save(self.cfg_path)
        
    def load_model(self):
        cfg = DinosrAudioConfig.load(self.cfg_path)
        dinosr = DinoSR(cfg, model_creator).to(device)
        dinosr.load(self.loss_model_path)
        with open(self.performance_path, 'r') as file:
            self.performance = yaml.load(file, Loader=yaml.FullLoader)
        self.current_step = self.performance['step']
        return dinosr

    def get_current_step(self):
        return self.current_step

# Define other necessary functions and classes
def save_training_history(history, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(history, file)