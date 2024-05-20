import json
from argparse import ArgumentParser
from mancalazero.train import train
from mancalazero.mancala import Mancala
from mancalazero.mancalanet import MancalaNet


parser = ArgumentParser()
parser.add_argument('config_json')
args = parser.parse_args()


with open(args.config_json) as f:
    config = json.load(f)


train(
    Mancala,
    MancalaNet,
    **config
)