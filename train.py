import argparse
import os
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("exp_name", help="path to experiment file")
args = parser.parse_args()

exp_name = args.exp_name
path = os.path.join('configs', exp_name)
path = path.replace('/', '.')

trainer = importlib.import_module(path).trainer
trainer.train()