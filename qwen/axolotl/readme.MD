# Steps to launch qwen fine-tuning 


## QWQ-32B

### Install axolotl 
These steps assume you are working inside a cuda or pytorch container from NGC registry. 

```
sudo apt update  
pip install --upgrade pip setuptools wheel 
pip3 uninstall torch torchvision torchaudio 
pip3 install packaging ninja 
cd /shared
git clone https://github.com/axolotl-ai-cloud/axolotl.git 
cd /shared/axolotl  
pip3 install --no-build-isolation -e '.[flash-attn,deepspeed]'

```

### Create train YAML 
Edit dataset and other parameters as desired. The config file assumes that axolotl repo is cloned at /shared directory. 

```
base_model: "Qwen/Qwen2.5-32B"
resume_from_checkpoint: 
load_in_8bit: false
load_in_4bit: false
strict: false
shuffle_merged_datasets: true
trust_remote_code:
bf16: auto
fp16:
tf32: false
seed: 42
model_init_kwargs:
  init_device: "meta"
  resize_token_embeddings_strategy: "mean"
chat_template: qwen_25
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca  
trust_remote_code: true
dataset_prepared_path: /shared/qwen/axolotl/data/last_run_prepared
hub_model_id:
sequence_len: 8192
pad_to_sequence_len: true
sample_packing: false
wandb_project: <enter-wandb-project>
wandb_entity: <enter-wandb-username>
wandb_watch:
wandb_name: qwen-32b-deepspeed-z3-axolotl
wandb_run_id: qwen-32b-deepspeed-z3-axolotl
wandb_log_model:
output_dir: /shared/qwen/axolotl/output

gradient_accumulation_steps: 1
micro_batch_size: 3
num_epochs: 1
warmup_steps: 10
learning_rate: 1e-5
max_steps: 100
eval_steps: 10
save_steps: 100
save_total_limit: 10
#auto_find_batch_size:
load_best_model_at_end: true
metric_for_best_model: "eval_loss"
greater_is_better: false

group_by_length: false
train_on_inputs: false


gradient_checkpointing: true
early_stopping_patience: 3
lr_scheduler: cosine
weight_decay: 0.01
max_grad_norm: 1.0
xformers_attention:
deepspeed: /shared/axolotl/deepspeed_configs/zero3_bf16_cpuoffload_all.json
flash_attention: true
optimizer: adamw_torch_fused
lr_scheduler: cosine

val_set_size: 0.05
do_eval: true

save_on_each_node: false
ddp_timeout: 300000
fsdp:
fsdp_config:

```

### Run the training job 

If you are working with multiple nodes, run the script on each node. 

```
#!sh
huggingface-cli login --token <enter-hf-token-here>

export MASTER_ADDR=<enter-address-or-ip-of-master>

export HF_HOME=/shared/hf_cache  # incase you want models to be cached in shared drive. ignore this if not. 
export WANDB_API_KEY=<enter-wandb-key-here>
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_MERGE_NICS=0

torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 1111 --rdzv_backend c10d  --rdzv_endpoint  "$MASTER_ADDR:6000" -m axolotl.cli.train /shared/qwen_train_32b.yaml

```
