import torch
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append("/anvme/workspace/v120bb18-unreflectanything")
from utilities.visualization import rgb, panelize

# %load_ext autoreload
# %autoreload 2

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    curr_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(curr_device)
    print(f"CUDA is available: {num_devices} device(s) detected.")
    print(f"Current device id: {curr_device} - {device_name}")
else:
    print("CUDA is not available")

from utilities.config import load_and_process_config,create_datasets_from_config
from utilities import tensor_dict_summarize
from tqdm import tqdm

config = load_and_process_config("/anvme/workspace/v120bb18-unreflectanything/configs/end2end.yaml")
config.DATASETS.ALL_DATASETS.SAMPLE_EVERY_N =1
config.DATASETS.ALL_DATASETS.TRAIN_SCENES =[""]
dataset = create_datasets_from_config(config)["Training"]

from typing import Any


dataset.return_metadata = True
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,num_workers=0)
paths = []
sizes = []

for n, batch in enumerate(tqdm(dataloader)):
    try:
        metadata = batch["metadata"]
        paths += metadata["raw_path"]
        sizes.append(metadata["orig_size"])
    except Exception as e:
        print(f"Error at batch {n}: {e}")
        print(f"Previous paths: {paths[-3:]}")
        continue
