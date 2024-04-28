import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils

import logging
log = logging.getLogger(__name__)

def get_components(config):
    model = instantiate(config.model)
    dataset = instantiate(config.dataset)
    return model, dataset

def run(config: DictConfig):
    utils.set_seed(config.exp.seed)
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # setup model and dataset
    model, dataset = get_components(config)

    # fit model
    model.train(dataset)

    # validate model
    model.validate(dataset)