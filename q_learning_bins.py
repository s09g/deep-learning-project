from gym import wrappers
import gym
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as import pd
from datetime import datetime

def build_state(features):
    return int("".join(map(lambda feature : str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[valie], bins=bins)[0]
