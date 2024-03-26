import time

from copy import deepcopy

from src.util import (
    get_many_random_hyperparams,
    load_defect_data,
    run_experiment,
)

import numpy as np

from tqdm import tqdm


hpo_space = {
    "learner": ["dt"],
    "preprocessor": ["normalize", "standardize", "minmax", "maxabs", "robust"],
    "wfo": [False, True],
    "smote": [False, True],
    "smooth": [False],
    "ultrasample": [False, True],
    "learner_params": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, .05, .1, .2],
        "min_samples_leaf": [1, 2, .05, .1, .2],
        "max_features": ["sqrt", "log2"]
    }
}

filename = "ivy"

perfs = []
start_time = time.time()
for _ in range(20):
    scores = []
    num_configs = 30
    data_orig = load_defect_data(filename)
    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in tqdm(configs):
        data = deepcopy(data_orig)
        print(f"Config: {config}")
        perf = run_experiment(data, 2, config)
        print("[main] Accuracy:", perf)
        scores.append(perf)

    perfs.append(max(scores))

print(f"Scores: {perfs}")
print(f"Median: {np.median(perfs)}")
print(f"Time: {time.time() - start_time}")
