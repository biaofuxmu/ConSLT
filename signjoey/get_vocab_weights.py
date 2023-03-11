#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
from torchtext.data import Dataset
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel
from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)

def get_vocab_weight(
    cfg_file, ckpt: str, vocab_file: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    _, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])
    print(ckpt)
    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    vocab_weight = model.decoder.output_layer.weight.cpu().detach().numpy()

    print('vocab_weight.shape')
    print(vocab_weight.shape)

    np.savetxt(vocab_file, vocab_weight)