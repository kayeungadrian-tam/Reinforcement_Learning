"""
Once a model is learned, use this to play it.
"""

import game
import numpy as np
from network import neural_net

NUM_SENSORS = 4


def play(model):
    myline = [[0,0],[10,10]]
    car_distance = 0
    game_state = game.Main(myline)

    # Do nothing to get initial.
    state, reward = game_state.step(action=0)    

    # Move.
    while True:
        car_distance += 1

        # Choose action.
        state_n = np.reshape(state, (1,NUM_SENSORS))
        action = (np.argmax(model.predict(state_n, batch_size=1)))

        # Take action.
        state, reward = game_state.step(action)
        game_state.render()
        # print(state)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    saved_model = '../saved-models/128-128-64-50000-2500.h5'
    model = neural_net(NUM_SENSORS, [128, 128], saved_model)
    play(model)
