import os, sys
import argparse
import torch

from connectomics.utils.system import init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer

from connectomics.config import get_cfg_defaults, save_all_cfg, update_inference_cfg
from connectomics.engine import Trainer
import torch.backends.cudnn as cudnn
import os

# os.environ['CUDA_VISIBLE_DEVICE'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str, default='/home/changs/pytorch_connectomics/configs/CS.yaml',
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--inference', default=True,
                        help='inference mode') # 当在终端运行的时候，如果不加入--inference, 那么程序运行时不执行预测（即为default false），如果加上了--inference,不需要指定True/False,程序运行时，都是true
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--checkpoint', type=str, default='/home/changs/pytorch_connectomics/dataset/output/checkpoint_50000.pth.tar',
                        help='path to load the checkpoint')
    parser.add_argument('--manual-seed', type=int, default=None)
    parser.add_argument('--local_world_size', type=int, default=1,
                        help='number of GPUs each process.')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='node rank for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='run the scripts in debug mode')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def main():
    # arguments
    args = get_args()
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    if args.local_rank == 0 or args.local_rank is None:
        # In distributed training, only print and save the configurations
        # using the node with local_rank=0.
        print("PyTorch: ", torch.__version__)
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # start training or inference
    mode = 'test' if args.inference else 'train'
    trainer = Trainer(cfg, device, mode,
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(
        args.local_rank, device))


if __name__ == "__main__":
    main()
