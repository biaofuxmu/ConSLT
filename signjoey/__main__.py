import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test
from signjoey.get_vocab_weights import get_vocab_weight

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")

print('----main')
def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "vocab"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument("--output_path", type=str, help="path for saving translation output")

    ap.add_argument("--vocab_file", type=str, help="path for saving vocabulary weight")

    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")

    args = ap.parse_args()

    if args.mode == "train":
        train(args=args)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    elif args.mode == "vocab":
        print('----vocab')
        get_vocab_weight(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path, vocab_file=args.vocab_file)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
