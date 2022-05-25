import matplotlib.pyplot as plt
import pickle
from argparse import Namespace
from typing import Dict
import numpy as np

from config import load_config
from utils import generate_data, create_model, set_up_logger, CommandLineHandler
from evaluation.evaluator import Evaluator
import torch
import random

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

def run_experiment(config: Dict, exp_name: str = ""):
    # generate data
    data = generate_data(config)

    # create requested model
    model = create_model(config)

    # set up logger
    logger = set_up_logger(exp_name=exp_name, config=config)

    # run experiment
    for update_ind, (points, occupancy, _) in enumerate(data):
        print(f"current update: {update_ind}")
        model.update(points, occupancy)
        if logger is not None:
            logger.update(data=np.vstack((points, occupancy)), model=model)

    # save model
    logger.save_model(model)  # save model to experiment file
    model.save(f"latest_model.pkl")  # save latest model for easy access

    return model, logger.exp_name


def main(args: Namespace):
    """ run / visualize / evaluate an LHM experiment """
    config = load_config()  # load configuration
    exp_name = args.exp_name

    # run experiment or load from pickle
    if args.load_model:
        with open(f"latest_model.pkl", "rb") as file:
            model = pickle.load(file)
    else:
        model, exp_name = run_experiment(config=config, exp_name=exp_name)

    # visualize results
    fig, ax = plt.subplots(nrows=1, ncols=1)
    mapping = model.plot(ax=ax, resolution=100)
    fig.colorbar(mapping)
    fig.show()

    # evaluate results
    evaluator = Evaluator(exp_name=exp_name)
    evaluator.evaluate_model()
    if args.gen_video:
        evaluator.create_update_video_stream_for_component()


if __name__ == "__main__":
    clh = CommandLineHandler()
    args = clh.parse()
    main(args=args)
