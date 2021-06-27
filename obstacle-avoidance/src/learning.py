import main
import numpy as np
import random
import csv
from network import neural_net, LossHistory
import os.path
import timeit

num_input = 3
gamma = 0.9
tuning = False



def train_net(model, params):
    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 2500  # Number of frames to play.
    max_car_distance = 0
    car_distance = 0
    batchSize = params['batchSize']
    buffer = params['buffer']

    game_state = main.Main()

    reward, state = game_state.step()

    start_time = timeit.default_timer()

    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').
    loss_log = []

    t = 0
    while t < train_frames:

        t += 1
        car_distance += 1

        # state = np.reshape(state, (state.shape[1],state.shape[0]))

        # print(state.shape)

        if t < 10:
            action = 0

        # Choose an action.
        elif random.random() < epsilon or t < observe:
            action = np.random.randint(-3, 3)  # random
        else:
            # Get Q values for each action.
            state_n = np.reshape(state, (1,num_input))
            qval = model.predict(state_n, batch_size=1)
            action = (np.argmax(qval))  # best


        reward, new_state = game_state.step(action, )

        replay.append((state, action, reward, new_state, ))

        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch2(minibatch, model)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

            # Train the model on this batch.
            history = LossHistory()
            model.fit(
                X_train, y_train, batch_size=batchSize,
                verbose=0, callbacks=[history]
            )
            loss_log.append(history.losses)

        state = new_state

        # Decrement epsilon over time.
        if epsilon > 0 and t > observe:
            epsilon -= (1.0/train_frames)

        # We died, so update stuff.
        if reward == -500:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (100*state[3], t, epsilon, car_distance, fps))

            # Reset.
            car_distance = 0

        # Save the model every 25,000 frames.
        if t % 500 == 0:
            model.save_weights('../saved-models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Saving model %s - %d" % (filename, t))



    log_results(filename, data_collect, loss_log)

def process_minibatch2(minibatch, model):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training FPS
    
    # instead of feeding data to the model one by one, 
    #   feed the whole batch is much more efficient

    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, num_input))
    actions = np.zeros(shape=(mb_len,))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, num_input))

    # print(actions)

    for i, m in enumerate(minibatch):
        # print(m)
        old_state_m, action_m, reward_m, new_state_m = m
        # print(action_m[1])
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]


    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)

    maxQs = np.max(new_qvals, axis=1)
    y = old_qvals
    non_term_inds = np.where(rewards != -500)[0]
    term_inds = np.where(rewards == -500)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (gamma * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    return X_train, y_train

def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our predicted best move.
        maxQ = np.max(newQ)
        y = np.zeros((1, 7))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (gamma * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(num_input,))
        y_train.append(y.reshape(7,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])

def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_net(num_input, params['nn'])
        train_net(model, params)
    else:
        print("Already tested.")



if __name__=="__main__":
    if tuning:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)


    else:
        nn_param = [128, 128]
        params = {
            "batchSize": 64,
            "buffer": 50000,
            "nn": nn_param
        }
        model = neural_net(num_input, nn_param)
        train_net(model, params)









