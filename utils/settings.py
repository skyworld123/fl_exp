import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_random_state():
    state = {
        'random': random.getstate(),
        'np.random': np.random.get_state(),
    }
    return state


def set_random_state(state: dict):
    random.setstate(state['random'])
    np.random.set_state(state['np.random'])
