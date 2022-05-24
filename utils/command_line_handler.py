from argparse import ArgumentParser


class CommandLineHandler:
    def __init__(self):
        # initialize parser
        self.parser = ArgumentParser()

        # add arguments to parser
        self.add_args()

    def add_args(self):
        self.parser.add_argument("--load_model", action="store_true")
        self.parser.add_argument("--exp_name", type=str, default="")

    def parse(self):
        return self.parser.parse_args()
