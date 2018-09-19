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

    task.run(frrn.model())


if __name__ == "__main__":
    main()
