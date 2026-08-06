import os
import sys

import torch
import pandas as pd

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = "/kaggle_simulations/agent"  # Kaggle exec()s the file so __file__ is undefined
_cg  = os.path.join(_dir, "cg-lib")
if not os.path.exists(_cg):
    _cg = "/kaggle_simulations/agent/cg-lib"
sys.path.insert(0, _cg)
sys.path.insert(0, _dir)

from agent_core import MyModel, mcts_agent  # auto-generated from RL_TRM2.ipynb by create_submission.py
from cg.api import Observation, to_observation_class

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_path = os.path.join(_dir, "model.pth")
if not os.path.exists(_model_path):
    _model_path = "/kaggle_simulations/agent/model.pth"
_model = MyModel(128, 2, 256, 3, 1)
_model.load_state_dict(torch.load(_model_path, map_location=_device, weights_only=True))
_model.to(_device).eval()


def read_deck() -> list[int]:
    file_path = "deck.xlsx"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.xlsx"
    return pd.read_excel(file_path, header=None).iloc[:, 0].tolist()


_deck = read_deck()


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial selection: return the 60-card deck.
        return _deck
    with torch.inference_mode():
        return mcts_agent(obs_dict, _deck, _model)[0]
