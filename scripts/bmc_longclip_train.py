import subprocess, threading
import os, time, random
import pandas as pd
import uuid
import socket

from typing import Optional
from datetime import datetime
from huggingface_hub import HfApi, HfFolder


#------------------------/  Parameters   /--------------------------

# Compute
NUM_NODES = 1
NUM_GPUS = 4
NUM_CPUS_PER_GPU = 8

# Job name
JOB_NAME = "bmc-long-512"
now = datetime.now()
datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")
EXPERIMENT_NAME = f"{JOB_NAME}-{datetime_str}"  # <-- use the same name for resuming

# Conda environment
CONDA_ENV = "train_clip"

# HF data
DATA_NAME = "biomedica_webdataset_24M"
HF_ID = "BIOMEDICA"
REPO_ID = f"{HF_ID}/{DATA_NAME}"
DATA_PATH = f"https://huggingface.co/datasets/{HF_ID}/{DATA_NAME}/resolve/main/"

# URL_PATH points to the file with training data urls
URL_PATH = f"../scripts/temp/data_urls/urls_{DATA_NAME}.txt"

# dir where checkpoint dir is created
LOG_PATH = "logs"
WANDB_PROJECT =  ""
WANDB_RUN_GROUP = ""

# Training 
TIME="100:00:00"
LEARNING_RATE = 5e-4
WARM_UP = 2000
BATCH_SIZE = 1024
EPOCH = 20
#-------------------------------------------------------------------


def generate_tar_urls_from_repo(base_path: str, repo_id: str, url_path: str) -> int:
    """
    Retrieve .tar filenames from a Hugging Face dataset repository, filter them,
    and generate a file with URLs for each .tar file in the repository. 

    Each URL is formatted by prepending the base path to the filename, and 
    URLs are joined with "::". The URLs are saved to a file named 'urls.txt'.

    Args:
        base_path (str): The base URL path to prepend to each filename.
        repo_id (str): The ID of the Hugging Face repository (e.g., "username/repo_name").

    Returns: number of tar files contained in the dataset repo
    """
    api = HfApi()
    token = HfFolder.get_token()
    
    # Retrieve the file list from the specified repository
    file_list = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    tar_files = [file for file in file_list if file.endswith('.tar')]
    len_tar_files = len(tar_files)
    
    # If no .tar files are found, return None
    if not tar_files:
        return None
    
    # Generate URLs and write them to a file
    urls = "::".join([f"{base_path}{filename}" for filename in tar_files])
    with open(url_path, "w") as url_file:
        url_file.write(urls)
    
    return len_tar_files


def find_free_port() -> int:
    """
    Find and return an available port on the local machine.

    This function creates a temporary socket, binds it to a free port assigned
    by the operating system, and then closes the socket. The assigned port 
    number is returned.

    Returns:
        int: A free port number available for use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# Function to submit and monitor a single job
def submit_job():
    
    os.makedirs('./temp/data_urls', exist_ok=True)
    os.makedirs('./temp/slurm_logs', exist_ok=True)
    os.makedirs('./temp/sh_script', exist_ok=True)

    # create url string text file
    len_tar = generate_tar_urls_from_repo(DATA_PATH, REPO_ID, URL_PATH)
    SAMPLE_SIZE = len_tar * 10000

    # RDZV_ID = str(random.randint(0, 9999)) # trunkate to 4 digits
    # RDZV_PORT= random.randint(20000, 29999)  # Range from 20000 to 29999
    RDZV_ID = str(uuid.uuid4())[:8]  # Truncate for readability
    RDZV_PORT = find_free_port()  # Get an available port

    BATCH_SCRIPT = f"./temp/sh_script/torchrun{RDZV_PORT}.sh"
    print("writing", BATCH_SCRIPT)

    with open(BATCH_SCRIPT, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes={NUM_NODES}
#SBATCH --ntasks={NUM_NODES}
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --time={TIME}
#SBATCH --mem=180gb
#SBATCH --cpus-per-task={NUM_CPUS_PER_GPU}
#SBATCH --output=./slurm_logs/base-%j-out.txt
#SBATCH --error=./slurm_logs/base-%j-err.txt

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export OMP_NUM_THREADS=1
export FI_PROVIDER=efa

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP={WANDB_RUN_GROUP}

source ~/.bashrc
conda activate {CONDA_ENV}

cd ../src

srun torchrun --nproc_per_node={NUM_GPUS} --nnodes={NUM_NODES} --node_rank=$SLURM_NODEID --rdzv_id {RDZV_ID} --rdzv_backend c10d --rdzv_endpoint $head_node_ip:{RDZV_PORT} -- \
	-m open_clip_train.main \
    --save-most-recent \
    --resume latest \
    --name {EXPERIMENT_NAME} \
    --train-data {URL_PATH} \
    --train-num-samples {SAMPLE_SIZE} \
    --lr-scheduler "cosine" \
    --dataset-type "webdataset" \
    --lr {LEARNING_RATE} \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup {WARM_UP} \
    --batch-size {BATCH_SIZE} \
    --epochs {EPOCH} \
    --workers 2 \
    --model "BioClinical-ModernBERT-large-ViT-L-14-pretrained" \
    --precision "fp32" \
    --local-loss \
    --grad-clip-norm 1.0 \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 1 \
    --seed 0 \
    --logs {LOG_PATH} \
    --report-to "wandb" \
    --wandb-project-name {WANDB_PROJECT}
""")

    # Submit the job
    sbatch_output = subprocess.check_output(["sbatch", BATCH_SCRIPT]).decode("utf-8")


if __name__=='__main__':
    submit_job()   