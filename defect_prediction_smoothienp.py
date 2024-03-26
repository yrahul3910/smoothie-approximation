import time

from copy import deepcopy

from src.util import (
    get_many_random_hyperparams,
    get_smoothness,
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
    data_orig = load_defect_data(filename)

    best_betas = []
    best_configs = []
    keep_configs = 5
    num_configs = 30

    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in tqdm(configs):
        data = deepcopy(data_orig)
        smoothness = get_smoothness(data, 2, config)

        if (len(best_betas) < keep_configs or smoothness < max(best_betas)) and smoothness > 0:
            best_betas.append(smoothness)
            best_configs.append(config)

            best_betas, best_configs = zip(
                *sorted(zip(best_betas, best_configs), reverse=False, key=lambda x: x[0])
            )
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

    scores = []
    for beta, config in zip(best_betas, best_configs):
        data = deepcopy(data_orig)
        print(f"Config: {config}\nbeta: {beta}")
        perf = run_experiment(data, 2, config)
        print("[main] Accuracy:", perf)
        scores.append(perf)

    perfs.append(max(scores))

print(f"Scores: {perfs}")
print(f"Median: {np.median(perfs)}")
print(f"Time: {time.time() - start_time}")
