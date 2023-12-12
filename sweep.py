import wandb
from main import main

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "train.rec_loss"},
    "parameters": {
        "learning_rate": {"max": 1e-3, "min": 5e-5, "distribution": "log_uniform_values"},
        "rec_loss": {"max": 1.0, "min": 0.01},
        "rec_loss_unnormalized": {"max": 1.0, "min": 0.01},
        "entropy_loss": {"min": 1e-4, "max": 1e0, "distribution": "log_uniform_values"},
        "commit_loss": {"min": 1e-3, "max": 1e0, "distribution": "log_uniform_values"},
    },
}

def run_main():
    wandb.init()
    main(
        preprocessed_dataset_path_or_url= "/hdd/laion2B-p14-0.012-partial-0-2500/00{0000..1386}.tar",
        torch_compile=False,
        seed=42,
        num_workers=2,
        grad_accumulation_steps=2,
        batch_size=8,
        train_norm_iters=0,
        max_iters= 4000,
        loss_fool_discriminator=0.15,
        log_every=500,
        model_config_path= "./conf/patch14-l.json",
        model_resume_path="./out/2023-12-10_01-39-44/model/",
        sample_patches_beta= 0.012,
        should_save=False,
        autocast_dtype='fp16',
        **wandb.config
    )

sweep_id = wandb.sweep(sweep=sweep_configuration, project="vq-experiments")
wandb.agent(sweep_id, function=run_main, count=30)
