import numpy as np
from tensorflow.keras.callbacks import Callback
from config import VALIDATION_INTERVAL
from utils import calculate_layer_width
from model import build_deep_halo

class RMSECallback(Callback):
    def __init__(self, choice_sets, empirical_dist, interval):
        super().__init__()
        self.choice_sets = choice_sets
        self.empirical_dist = empirical_dist
        self.interval = interval
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0: return
        errors = []
        for cs in self.choice_sets:
            inp = np.zeros((1, self.model.input_shape[1]))
            inp[0, list(cs)] = 1.0
            pred = self.model.predict(inp, verbose=0)[0]
            errors.append(np.mean(np.square(pred[list(cs)] - self.empirical_dist[cs])))
        rmse = np.sqrt(np.mean(errors))
        self.history.append(rmse)
        print(f" - Epoch {epoch+1} RMSE: {rmse:.6f}")

def run_experiment(dataset, empirical_dist, choice_sets, config):
    results = {}
    # Extract training data from ChoiceDataset
    # In featureless mode, the 'availability' is our primary input
    x_train = dataset.available_items_by_choice
    # One-hot encoding choices for MSE loss
    y_train = np.zeros((len(dataset), config['universe_size']))
    for i, c in enumerate(dataset.choices):
        y_train[i, int(c)] = 1.0

    for budget in config["param_budgets"]:
        results[budget] = {}
        for depth in config["network_depths"]:
            width = calculate_layer_width(depth, budget, config["universe_size"])
            if width <= 0: continue
            
            model = build_deep_halo(config["universe_size"], width, depth)
            model.compile(optimizer='adam', loss='mse')
            
            cb = RMSECallback(choice_sets, empirical_dist, VALIDATION_INTERVAL)
            model.fit(x_train, y_train, batch_size=config['batch_size'], 
                      epochs=config['epochs'], callbacks=[cb], verbose=0)
            
            results[budget][depth] = {'history': cb.history, 'final': cb.history[-1]}
    return results        

def run_experiment(dataset, empirical_dist, choice_sets, config):
    results = {}
    # Extract training data from ChoiceDataset
    # In featureless mode, the 'availability' is our primary input
    x_train = dataset.available_items_by_choice
    # One-hot encoding choices for MSE loss
    y_train = np.zeros((len(dataset), config['universe_size']))
    for i, c in enumerate(dataset.choices):
        y_train[i, int(c)] = 1.0

    for budget in config["param_budgets"]:
        results[budget] = {}
        for depth in config["network_depths"]:
            width = calculate_layer_width(depth, budget, config["universe_size"])
            if width <= 0: continue
            
            model = build_deep_halo(config["universe_size"], width, depth)
            model.compile(optimizer='adam', loss='mse')
            
            cb = RMSECallback(choice_sets, empirical_dist, VALIDATION_INTERVAL)
            model.fit(x_train, y_train, batch_size=config['batch_size'], 
                      epochs=config['epochs'], callbacks=[cb], verbose=0)
            
            results[budget][depth] = {'history': cb.history, 'final': cb.history[-1]}
    return results