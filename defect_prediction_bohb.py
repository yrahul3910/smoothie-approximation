import time

from copy import deepcopy

from src.util import load_defect_data, run_experiment

import bohb.configspace as cs

from bohb import BOHB


hpo_space = cs.ConfigurationSpace([
    cs.CategoricalHyperparameter("preprocessor", ["normalize", "standardize", "minmax", "maxabs", "robust"]),
    cs.CategoricalHyperparameter("wfo", [False, True]),
    cs.CategoricalHyperparameter("smote", [False, True]),
    cs.CategoricalHyperparameter("ultrasample", [False, True]),

    cs.CategoricalHyperparameter("criterion", ["gini", "entropy"]),
    cs.CategoricalHyperparameter("max_depth", [None, 2, 3, 4, 5]),
    cs.CategoricalHyperparameter("min_samples_split", [2, .05, .1, .2]),
    cs.CategoricalHyperparameter("min_samples_leaf", [1, 2, .05, .1, .2]),
    cs.CategoricalHyperparameter("max_features", ["sqrt", "log2"])
])

filename = "camel"

perfs = []
start_time = time.time()
for _ in range(20):
    scores = []
    num_configs = 30
    data_orig = load_defect_data(filename)

    def objective(config, *args, **kwargs):
        data = deepcopy(data_orig)

        formatted_config = {
            "preprocessor": config["preprocessor"],
            "wfo": config["wfo"],
            "smote": config["smote"],
            "smooth": False,
            "ultrasample": config["ultrasample"],
            "learner": "dt",
            "learner_params": {
                "criterion": config["criterion"],
                "max_depth": config["max_depth"],
                "min_samples_split": config["min_samples_split"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"]
            }
        }

        print(f"Config: {formatted_config}")
        perf = run_experiment(data, 2, formatted_config)
        print("[main] Accuracy:", perf)
        scores.append(perf)
        return perf[0]

    bohb = BOHB(configspace=hpo_space, evaluate=objective, min_budget=1, max_budget=30)
    logs = bohb.optimize()
    perfs.append(max(scores))
