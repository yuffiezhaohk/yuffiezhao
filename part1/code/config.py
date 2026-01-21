# CONFIG = {
#     "universe_size": 20,
#     "choice_set_size": 15,
#     "obs_per_set": 80,
#     "epochs": 500,
#     "learning_rate": 1e-4,
#     "batch_size": 1024,
#     "param_budgets": [200000, 500000],
#     "network_depths": [3, 4, 5, 6, 7]
# }
# DATA_FILENAME = "full_scale_data_final.npz"
# VALIDATION_INTERVAL = 10
# SEED = 42

CONFIG = {
    "universe_size": 20,
    "choice_set_size": 15,
    "obs_per_set": 80,
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "param_budgets": [500, 1000],
    "network_depths": [2,3]
}
DATA_FILENAME = "debug_data_final.npz"
VALIDATION_INTERVAL = 1
SEED = 42

