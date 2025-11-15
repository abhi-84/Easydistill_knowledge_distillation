import json

# Base config template
base_config = {
  "job_type": "kd_white_box",
  "dataset": {
    "instruction_path": "",  # Will be filled per chunk
    "labeled_path": "",       # Will be filled per chunk
    "logits_path": "",        # Will be filled per chunk
    "template": "./easydistill/configs/chat_template/chat_template_kd.jinja",
    "seed": 42
  },
  "inference": {
    "enable_chunked_prefill": True,
    "seed": 777,
    "gpu_memory_utilization": 0.9,
    "temperature": 0.8,
    "trust_remote_code": True,
    "enforce_eager": False,
    "max_model_len": 512,
    "max_new_tokens": 400,
    "top_logits_num": 20
  },
  "distillation": {
    "kd_ratio": 0.5,
    "max_seq_length": 512,
    "distillation_type": "forward_kld"
  },
  "models": {
    "teacher": "./easydistill/teacher/Qwen/Qwen2.5-7B-Instruct",
    "student": "./easydistill/result_stage1/checkpoint-4000"
  },
  "training": {
    "output_dir": "./easydistill/result_stage2/",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "save_steps": 2000,
    "logging_steps": 1,
    "learning_rate": 2e-05,
    "weight_decay": 0.05,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine"
  }
}

# Create config for each chunk
config_dir = './easydistill/recipes/distilqwen_series/distillqwen2.5/'
chunks_dir = './easydistill/chunks/'

for i in range(5):
    config = base_config.copy()
    config['dataset'] = base_config['dataset'].copy()

    # Set paths for this chunk
    config['dataset']['instruction_path'] = f'{chunks_dir}distilqwen-chunk-{i}.json'
    config['dataset']['labeled_path'] = f'{chunks_dir}distilqwen-chunk-{i}.json'
    config['dataset']['logits_path'] = f'{chunks_dir}logits-chunk-{i}.json'

    # Save config
    config_file = f'{config_dir}stage2_chunk_{i}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f" Created config: stage2_chunk_{i}.json")

print("\nAll config files created!")
