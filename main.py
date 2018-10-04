import keras

from utils.config_loader import ConfigLoader
from model.frrn import FULL_RESOLUTION_RESIDUAL_NETWORKS
from tasks import TASKS

from argparse import ArgumentParser


def parse_cli():
    parser = ArgumentParser(description="FRRN")
    parser.add_argument("--config", default="config.yaml", help="path to the yaml config file")
    parser.add_argument("task", choices=TASKS.keys())
    args = parser.parse_args()

    return args


def main():
    args = parse_cli()

    config_loader = ConfigLoader()
    if not config_loader.load(args.config):
        print("Could not load config!")
        exit(-1)

    config = config_loader.getConfig()

    # now build from config
    frrn = FULL_RESOLUTION_RESIDUAL_NETWORKS[config["model"]["architecture"]](**config["model"])
    frrn.model().summary()

    # load pretrained weights if available
    if config.has_key("weights"):
        frrn.model().load_weights(config["weights"])

    # now run one of the registered tasks e.g. training or prediction
    task = TASKS[args.task](config[args.task])

    import datetime
    log_dir = "./logs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    import os
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    elif task == 'train':
        print("The log directory already exists!")
        exit(-1)

    from shutil import copy2
    if config.has_key("weights"):
        copy2(config["weights"], os.path.join(log_dir, "base_model.h5"))

    # copy the yaml config to the log directory to know how to reproduce results
    copy2(args.config, os.path.join(log_dir, 'config.yaml'))

    task.run(frrn.model(), log_dir)


if __name__ == "__main__":
    main()
