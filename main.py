import argparse

from training import Trainer
from modeling import MaskedAutoencoder


def main(args):

    model = MaskedAutoencoder()

    if args.train:
        trainer = Trainer(model)
        trainer.train()
    elif args.test:
        pass

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--ckpt")

    # args = parser.parse_args()
    # main(args)
    model = MaskedAutoencoder()
    trainer = Trainer(model)
    trainer.train()
