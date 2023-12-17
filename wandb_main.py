import wandb
from main import main

def run_main():
    wandb.init()
    main(
        preprocessed_dataset_path_or_url= "/hdd/laion2B-p14-0.012-partial-0-2500/00{0000..1386}.tar",
        torch_compile=True,
        seed=420,
        num_workers=2,
        grad_accumulation_steps=1,
        batch_size=32,
        train_norm_iters=10,
        max_iters= 4000,
        log_every=500,
        model_config_path= "./conf/patch14-l.json",
        sample_patches_beta=0.012,
        should_save=False,
        autocast_dtype='no',
        model_resume_path="./out/clip_merged_model/",
        **wandb.config
    )

run_main()
