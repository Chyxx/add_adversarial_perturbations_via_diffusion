import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--class_count", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--alpha", type=float, default=1e-3, help="coefficient of l1-loss")
parser.add_argument("--beta", type=float, default=1, help="coefficient of l2-loss")
parser.add_argument("--gamma", type=float, default=3e-1, help="coefficient of ssim-loss")
parser.add_argument("--delta", type=float, default=1.5e-4, help="step of estimated gradient to add")
parser.add_argument("--epsilon", type=float, default=1e-2, help="std of generated noise")
parser.add_argument("--sigma", type=float, default=1e-4, help="std of extra noise")
parser.add_argument("--num", type=int, default=20, help="number of try noise")
parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--model_channels", type=int, default=64, help="number of model channels")
parser.add_argument("--processor_model", type=str, default="unet")
parser.add_argument("--trier_model", type=str, default="try_by_svd")
parser.add_argument("--detector_model", type=str, default="resnet34")
parser.add_argument("-f", "--file", default="E:/data/imagenet_ai_0424_sdv5/train", type=str, help="path to data directory")
parser.add_argument("--processor_path", type=str, default="")
parser.add_argument("--detector_path", type=str, default="data/classifier-ckpt/resnet34/best_epoch.pkl")
parser.add_argument("--processor_save_path", type=str, default="data/processor-ckpt")
parser.add_argument("--detector_save_path", type=str, default="data/classifier-ckpt")
parser.add_argument("--steps", type=int, default=5, help="step number of diffusion")
opt = parser.parse_args()

