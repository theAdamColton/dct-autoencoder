import wandb
from main import main
import torch
import gc

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "train.rec_loss_unnormalized"},
    "parameters": {
        "learning_rate": {"max": 1e-3, "min": 5e-5, "distribution": "log_uniform_values"},
        "rec_loss": {"max": 1.0, "min": 0.01},
        "entropy_loss": {"min": 1e-5, "max": 1e0, "distribution": "log_uniform_values"},
        "commit_loss": {"min": 1e-5, "max": 1e0, "distribution": "log_uniform_values"},
    },
    #"early_terminate": {"type": "hyperband", "max_iter": 1000, "s": 10},
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="vq-experiments")

def run_main():
    wandb.init("vq-experiments")
    main(
        preprocessed_dataset_path_or_url= "/hdd/laion-improved-aesthetics-6p-preprocessed-p32-a0.015/000{000..199}.tar",
        torch_compile= True,
        seed= 42,
        num_workers=  2,
        grad_accumulation_steps= 2,
        batch_size=16,
        train_norm_iters= 10,
        max_iters= 50,
        model_config_path= "./conf/patch32-xl.json",
        model_resume_path="./out/clip_merged_model/",
        sample_patches_beta= 0.015,
        rec_loss_unnormalized=  1.0,
            **wandb.config)

wandb.agent(sweep_id, function=run_main, count=15)
