import torch

from models import *
# from attacks import linear_search_blended_uniform_noise_attack

CLASSIFICATION_TYPE = "one_to_all"  # or "one_to_one"

if CLASSIFICATION_TYPE == "one_to_all":
    OUTPUT_DIM = 2
    LABELS = None
else:
    OUTPUT_DIM = 11
    LABELS = {
        'Benign': 0,
        'FTP-BruteForce': 1,
        'SSH-Bruteforce': 2,
        'Infilteration': 3,
        'DDoS attacks-LOIC-HTTP': 4,
        'Brute Force -Web': 5,
        'Brute Force -XSS': 6,
        'SQL Injection': 7,
        'Bot': 8,
        'DoS attacks-SlowHTTPTest': 9,
        'DoS attacks-Hulk': 10
    }

# labels to integer mapping

BENIGN_ARG = 0  # the index of the benign class in the output of the model

TRAIN_MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN = False  # set to False to load the model instead of training it
# load_from_csv = False  # set to False to load the data from scratch
SPLIT_RATIO = 0.5  # proportion of the dataset to include in the train and validation split

# target model class
TARGET_MODEL_CLASS = MLP
ATTACK_BATCH_SIZE = 5000
# ATTACK = linear_search_blended_uniform_noise_attack