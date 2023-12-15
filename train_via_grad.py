"""
Train a diffusion model on images.
"""
import argparse
import os

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

from utils.dataset import tiny_loader
from config import opt
from classifier import Classifier
from visdom import Visdom


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    vis = Visdom()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion_esi, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion_esi)

    logger.log("creating data loader...")
    train_loader, val_loader = tiny_loader(args.batch_size)

    logger.log("training...")
    TrainLoop(
        vis=vis,
        model=model,
        diffusion_esi=diffusion_esi,
        diffusion=diffusion,
        classifier=Classifier(opt).to(dist_util.dev()),
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interaval,
        n_epochs=args.n_epochs,
        target_label=args.target_label,
        resume_checkpoint=args.resume_checkpoint,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=500,
        val_interaval=500,
        n_epochs=200,
        target_label=None,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()