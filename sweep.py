import wandb
from main import main

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "train.rec_loss_unnormalized"},
    "parameters": {
        "learning_rate": {"max": 1e-3, "min": 5e-5, "distribution": "log_uniform_values"},
        "rec_loss": {"max": 1.0, "min": 0.01},
        "rec_loss_unnormalized": {"max": 1.0, "min": 0.01},
        "entropy_loss": {"min": 1e-4, "max": 1e0, "distribution": "log_uniform_values"},
        "commit_loss": {"min": 1e-4, "max": 1e0, "distribution": "log_uniform_values"},
    },
    "early_terminate": {"type": "hyperband",},
}

def run_main():
    wandb.init()
    main(
        preprocessed_dataset_path_or_url= "/hdd/laion-improved-aesthetics-6p-preprocessed-p32-a0.007/000{000..430}.tar",
        torch_compile=True,
        seed= 42,
        num_workers=  2,
        grad_accumulation_steps=2,
        batch_size=6,
        train_norm_iters=20,
        max_iters= 3000,
        log_every=250,
        model_config_path= "./conf/patch32-xs.json",
        accelerator_resume_path="./out/2023-12-04_11-43-09/accelerator_state/",
        sample_patches_beta= 0.007,
        **wandb.config
    )

sweep_id = wandb.sweep(sweep=sweep_configuration, project="vq-experiments")
wandb.agent(sweep_id, function=run_main, count=30)
