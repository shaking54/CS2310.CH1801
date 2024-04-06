from pathlib import Path
import math
ORIGINAL_BATCH_SIZE = 8
DESIRED_BATCH_SIZE = 64
def get_config():
    return {
        "batch_size": DESIRED_BATCH_SIZE,
        "num_epochs": 20,
        # https://arxiv.org/pdf/1711.00489.pdf
        "lr": 10**-4 * math.sqrt(DESIRED_BATCH_SIZE/ORIGINAL_BATCH_SIZE),
        "seq_len": 128,
        "d_model": 512,
        "datasource": 'harouzie/vi_en-translation',
        "lang_src": "Vietnamese",
        "lang_tgt": "English",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/vi-en-translation",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
