import random

import numpy as np
import pandas as pd

from keras.utils import to_categorical
from raise_utils.data import Data, DataLoader
from raise_utils.hooks import Hook
from raise_utils.learners import Autoencoder
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Dataset filenames
defect_file_dic = {
    "ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],
    "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],
    "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],
    "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],
    "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"],
    "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"],
    "jedit": [
        "jedit-3.2.csv",
        "jedit-4.0.csv",
        "jedit-4.1.csv",
        "jedit-4.2.csv",
        "jedit-4.3.csv",
    ],
    "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
    "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"],
    "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"],
}


def load_defect_data(dataset: str):
    def _binarize(_, y):
        y[y > 1] = 1

    base_path = "../../DODGE Data/defect/"
    data = DataLoader.from_files(
        base_path=base_path,
        files=defect_file_dic[dataset],
        hooks=[Hook("binarize", _binarize)],
    )

    return data


def run_experiment(
    data: Data,
    n_class: int,
    config: dict
) -> list:
    print("[run_experiment] Getting model")
    model, data = get_model(
        data,
        n_class,
        config
    )

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    model.fit(data.x_train, data.y_train)

    if n_class == 2:
        y_pred = (model.predict(data.x_test) > 0.5).astype("int32")
    else:
        y_pred = np.argmax(model.predict(data.x_test), axis=-1)

    if n_class > 2 and len(data.y_test) > 1:
        data.y_test = np.argmax(data.y_test, axis=1)

    metrics = ClassificationMetrics(data.y_test, y_pred)
    metrics.add_metrics(["f1"])
    return metrics.get_metrics()


def loss(X: np.ndarray, y, model, learner: str) -> float:
    """
    Returns the negative log-liklihood of the model.
    """
    return log_loss([y], model.predict_proba(X.reshape(1, -1)), labels=[0, 1])


def get_smoothness(
    data: Data,
    n_class: int,
    config: dict
) -> float:
    model, data = get_model(data, n_class, config)

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    model.fit(data.x_train, data.y_train)

    betas = []
    for _ in range(30):
        px, py = random.choice(list(zip(data.x_train, data.y_train)))

        epsilon = .01
        ranges = np.max(data.x_train, axis=0) - np.min(data.x_train, axis=0)
        qx = px + epsilon * ranges
        qy = py

        def loss_fn(X, y):
            return loss(X, y, model, config["learner"])

        h = np.linalg.norm(px - qx)
        fx = loss_fn(px, py)  # f(x)
        f_xph = loss_fn(qx, qy)  # f(x + h)
        f_xmh = loss_fn(2 * px - qx, py)  # f(x - h)

        betas.append((f_xph + f_xmh - 2 * fx) / h**2)

    return np.median(betas)


def get_random_hyperparams(options: dict) -> dict:
    """
    Get hyperparameters from options.
    """
    hyperparams = {"learner_params": {}}
    for key, value in options.items():
        if key == "learner_params":
            for k, v in value.items():
                if isinstance(v, list):
                    hyperparams[key][k] = random.choice(v)
                elif isinstance(v, tuple):
                    hyperparams[key][k] = random.randint(v[0], v[1])
        else:
            if isinstance(value, list):
                hyperparams[key] = random.choice(value)
            elif isinstance(value, tuple):
                hyperparams[key] = random.randint(value[0], value[1])
    return hyperparams


def get_many_random_hyperparams(options: dict, n: int) -> list[dict]:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def remove_labels_legacy(data: Data) -> Data:
    # "Remove" labels
    lost_idx = np.random.choice(
        len(data.y_train),
        size=int(len(data.y_train) - np.sqrt(len(data.y_train))),
        replace=False,
    )

    x_lost = data.x_train[lost_idx]
    x_rest = np.delete(data.x_train, lost_idx, axis=0)
    y_lost = data.y_train[lost_idx]
    y_rest = np.delete(data.y_train, lost_idx, axis=0)

    if len(x_lost.shape) == 1:
        x_lost = x_lost.reshape(1, -1)
    if len(x_rest.shape) == 1:
        x_rest = x_rest.reshape(1, -1)

    # Impute data
    tree = KDTree(x_rest)
    _, idx = tree.query(x_lost, k=int(np.sqrt(np.sqrt(len(x_rest)))), p=1)
    y_lost = mode(y_rest[idx], axis=1)[0]
    y_lost = y_lost.reshape((y_lost.shape[0], y_lost.shape[-1]))

    assert len(x_lost) == len(y_lost)

    data.x_train = np.concatenate((x_lost, x_rest), axis=0)
    data.y_train = np.concatenate((y_lost, y_rest), axis=0)
    return data


def get_model(
    data: Data,
    n_class: int,
    config: dict
) -> tuple:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    """
    transform = Transform(config["preprocessor"])
    transform.apply(data)

    if config["smooth"]:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        if n_class == 2:
            smoother = Transform("smooth")
            smoother.apply(data)
        else:
            data = remove_labels_legacy(data)

    if config["ultrasample"]:
        # Apply WFO
        transform = Transform("wfo")
        transform.apply(data)

        # Reverse labels
        data.y_train = 1.0 - data.y_train
        data.y_test = 1.0 - data.y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5, verbose=0)
            ae.set_data(*data)
            ae.fit()

            loss = ae.model.history.history["loss"][-1]

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    learner_map = {
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "svm": SVC,
    }
    model = learner_map[config["learner"]](**config["learner_params"])
    data.y_train = data.y_train.astype(float)

    data.x_train, data.x_test, data.y_train, data.y_test = map(
        np.array, (data.x_train, data.x_test, data.y_train, data.y_test)
    )
    data.y_train = data.y_train.squeeze()
    data.y_test = data.y_test.squeeze()

    if config["wfo"]:
        transform = Transform("wfo")
        transform.apply(data)

    if config["smote"]:
        transform = Transform("smote")

        if n_class > 2 and len(data.y_train.shape) > 1:
            data.y_train = np.argmax(data.y_train, axis=1)

        transform.apply(data)

        if n_class > 2 and len(data.y_train.shape) == 1:
            data.y_train = to_categorical(data.y_train, n_class)

    return model, data


def split_data(filename: str, data: Data, n_classes: int) -> Data:
    if n_classes == 2:
        if filename == "firefox.csv":
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == "chromium.csv":
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0, np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(
                data.y_train < 3,
                1,
                np.where(data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4)),
            ),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(
                data.y_test < 3,
                1,
                np.where(data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4)),
            ),
        )
    elif n_classes == 7:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(
                data.y_train < 2,
                1,
                np.where(
                    data.y_train < 3,
                    2,
                    np.where(
                        data.y_train < 6,
                        3,
                        np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6)),
                    ),
                ),
            ),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(
                data.y_test < 2,
                1,
                np.where(
                    data.y_test < 3,
                    2,
                    np.where(
                        data.y_test < 6,
                        3,
                        np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6)),
                    ),
                ),
            ),
        )
    else:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(
                data.y_train < 2,
                1,
                np.where(
                    data.y_train < 3,
                    2,
                    np.where(
                        data.y_train < 4,
                        3,
                        np.where(
                            data.y_train < 6,
                            4,
                            np.where(
                                data.y_train < 8,
                                5,
                                np.where(
                                    data.y_train < 11,
                                    6,
                                    np.where(data.y_train < 21, 7, 8),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(
                data.y_test < 2,
                1,
                np.where(
                    data.y_test < 3,
                    2,
                    np.where(
                        data.y_test < 4,
                        3,
                        np.where(
                            data.y_test < 6,
                            4,
                            np.where(
                                data.y_test < 8,
                                5,
                                np.where(
                                    data.y_test < 11,
                                    6,
                                    np.where(data.y_test < 21, 7, 8),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    if n_classes > 2:
        binarizer = LabelBinarizer()
        data.y_train = binarizer.fit_transform(data.y_train)
        data.y_test = binarizer.transform(data.y_test)

    return data


def load_issue_lifetime_prediction_data(filename: str, n_classes: int) -> Data:
    df = pd.read_csv(f"./data/{filename}.csv")
    df.drop(["Unnamed: 0", "bugID"], axis=1, inplace=True)
    _df = df[["s1", "s2", "s3", "s4", "s5", "s6", "s8", "y"]]
    _df["s70"] = df["s7"].apply(lambda x: eval(x)[0])
    _df["s71"] = df["s7"].apply(lambda x: eval(x)[1])
    _df["s72"] = df["s7"].apply(lambda x: eval(x)[2])
    _df["s90"] = df["s9"].apply(lambda x: eval(x)[0])
    _df["s91"] = df["s9"].apply(lambda x: eval(x)[1])

    if filename == "firefox":
        _df["s92"] = df["s9"].apply(lambda x: eval(x)[2])

    x = _df.drop("y", axis=1)
    y = _df["y"]

    data = Data(*train_test_split(x, y))
    data = split_data(filename, data, n_classes)
    return data
