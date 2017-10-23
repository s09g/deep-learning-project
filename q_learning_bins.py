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

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velovity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velovity_bins = np.linspace(-3.5, 3.5, 9)
    
    def transform(self, parameter_list):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velovity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velovity_bins),
        ])

class Model:
    def __init__(self, parameter_list):
        pass
    
    def predict(self, parameter_list):
        pass