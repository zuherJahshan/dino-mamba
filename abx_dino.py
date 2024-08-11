from model_persistant_state import ModelPersistantState
from dataset import get_dataloader
import torch
import numpy as np
import sys
import os
import argparse

# use the argparser to get the following arguments:
'''
1. a requried argument for the model name
2. an optional argument that states weather results should be computed, or only a benchmark should be run
'''

argparser = argparse.ArgumentParser()
argparser.add_argument('model_name', type=str, help='Name of the model to load')
argparser.add_argument('--compute_results', action='store_true', help='Compute the results')
args = argparser.parse_args()

if __name__ == "__main__":
    # 1. load a model
    # get the model name as an argument
    model_name = args.model_name
    compute_results = args.compute_results
    # Check if model existing
    if not os.path.exists(f"models/{model_name}"):
        print(f"Model {model_name} does not exist")
        sys.exit(1)
    model_path = f"models/{model_name}"
    results_path = f"results/{model_name}/abx/"
    # remove already existing results
    os.system(f"rm -rf {results_path}")

    # initialize the submission
    os.system(f"zrc submission:init abxLS {results_path}")

    # Load the model
    model_persistant_state = ModelPersistantState(model_path)
    model = model_persistant_state.load_model()

    # 2. load a dataset
    dataloader = get_dataloader(20, data_type="abx")

    # generate results and write them to files
    if compute_results:
        print("Computing results...")
        for idx, (waveforms, lens, path) in enumerate(dataloader):
            model.eval()
            with torch.no_grad():
                result = model(waveforms, lens, only_student=True)
                result = result["student"]
            paths = [results_path + ".".join("/".join(p.split("/")[-2:]).split(".")[:-1]) + ".npy" for p in path]
            for batch_idx, path in enumerate(paths):
                # write tensor to the npy file in numpy format
                result_to_save = result[batch_idx].cpu().numpy()
                # transform to float64
                result_to_save = result_to_save.astype(np.float64)[:lens[batch_idx]]
                np.save(path, result_to_save)
            # print the progress in precentage
            print(f"\r{idx} results of {len(dataloader)} generated", end="")
        print("\nDone generating results, running benchmark")
    else:
        print("skipping results computation, applying benchmark only")

    # run the ABX evaluation
    os.system(f"zrc benchmarks:run abxLS {results_path}")