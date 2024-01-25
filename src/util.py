import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import mode
from raise_utils.transforms import Transform
from raise_utils.learners import Autoencoder
from raise_utils.data import Data, DataLoader
from raise_utils.hooks import Hook
from raise_utils.metrics import ClassificationMetrics
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss

from src.fastmap import Fastmap


# Dataset filenames
defect_file_dic = {'ivy':     ['ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv'],
            'lucene':  ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
            'poi':     ['poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
            'synapse': ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],
            'velocity': ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
            'camel': ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
            'jedit': ['jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv'],
            'log4j': ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
            'xalan': ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
            'xerces': ['xerces-1.2.csv', 'xerces-1.3.csv', 'xerces-1.4.csv']
            }


def load_defect_data(dataset: str):
    def _binarize(x, y): y[y > 1] = 1
    base_path = '../DODGE Data/defect/'
    data = DataLoader.from_files(
        base_path=base_path, files=defect_file_dic[dataset], hooks=[Hook('binarize', _binarize)])
    
    return data


def run_experiment(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool, n_units: int,
    n_layers: int, preprocessor: str) -> list:
    print('[run_experiment] Getting model')
    model, data = get_model(data, n_class, wfo, smote, ultrasample, smooth, n_units, n_layers, preprocessor)

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)
    print('[run_experiment] Got model')
    model.fit(data.x_train, data.y_train, epochs=100, verbose=1, batch_size=128)
    print('[run_experiment] Fit model')

    if n_class == 2:
        y_pred = (model.predict(data.x_test) > 0.5).astype('int32')
    else:
        y_pred = np.argmax(model.predict(data.x_test), axis=-1)
    
    if n_class > 2 and len(data.y_test) > 1:
        data.y_test = np.argmax(data.y_test, axis=1)

    metrics = ClassificationMetrics(data.y_test, y_pred)
    metrics.add_metrics(['accuracy'])
    return metrics.get_metrics()


data_kdtree = None


def loss(data: Data, sample: np.ndarray, model) -> float:
    """
    Compute the loss of the model on the given data.
    """
    global data_kdtree

    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    pred = model.predict_proba(sample)

    # Find closest point in data and grab its y
    if data_kdtree is None:
        data_kdtree = KDTree(data.x_train)

    _, idx = data_kdtree.query(sample, k=1)
    y = data.y_train[idx]

    return log_loss(y, pred)


def get_smoothness(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool, learner: str, preprocessor: str) -> float:
    model, data = get_model(data, n_class, wfo, smote, ultrasample, smooth, learner, preprocessor)

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    # Fit on 10n random samples
    idx = np.random.choice(len(data.x_train), size=10 * data.x_train.shape[1], replace=False)
    x_sample = data.x_train[idx]
    y_sample = data.y_train[idx]
    model.fit(x_sample, y_sample)
    
    tree = Fastmap(data.x_train)._recurse()
    betas = []
    stack = []
    
    # Recurse down the tree
    stack.append(tree)
    while len(stack) > 0:
        node = stack.pop()
        
        if node.left is None and node.right is None:
            # Compute beta
            parent = node.parent
            left_sv = parent.left_sv
            right_sv = parent.right_sv

            def loss_fn(x): 
                return loss(data, x, model)

            h = np.linalg.norm(left_sv - right_sv)
            fx = loss_fn(left_sv)  # f(x)
            f_xph = loss_fn(right_sv)  # f(x + h)
            f_xmh = loss_fn(left_sv - right_sv)  # f(x - h)
            f_x2h = loss_fn(2 * right_sv - left_sv)  # f(x + 2h)

            # Add both betas to the list
            betas.append((f_xph + f_xmh - 2 * fx) / h ** 2)
            betas.append((f_x2h + fx - 2 * f_xph) / h ** 2)

        if node.left is not None and node.right is not None:
            stack.append(node.left)
            stack.append(node.right)

    return max(betas)


def get_random_hyperparams(options: dict) -> dict:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            hyperparams[key] = random.randint(value[0], value[1])
    return hyperparams


def get_many_random_hyperparams(options: dict, n: int) -> list:
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
        len(data.y_train), size=int(len(data.y_train) - np.sqrt(len(data.y_train))), replace=False)

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
    _, idx = tree.query(x_lost, k = int(np.sqrt(np.sqrt(len(x_rest)))), p=1)
    y_lost = mode(y_rest[idx], axis=1)[0]
    y_lost = y_lost.reshape((y_lost.shape[0], y_lost.shape[-1]))

    assert len(x_lost) == len(y_lost)

    data.x_train = np.concatenate((x_lost, x_rest), axis=0)
    data.y_train = np.concatenate((y_lost, y_rest), axis=0)
    return data


def get_model(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool,
              learner: str, preprocessor: str) -> tuple:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    """
    transform = Transform(preprocessor)
    transform.apply(data)

    if smooth:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        print('[get_model] Running smooth')
        if n_class == 2:
            smoother = Transform('smooth')
            smoother.apply(data)
        else:
            data = remove_labels_legacy(data)
        print('[get_model] Finished running smooth')

    if ultrasample:
        # Apply WFO
        print('[get_model] Running ultrasample:wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running ultrasample:wfo')

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(*data)
            print('[get_model] Fitting autoencoder')
            ae.fit()
            print('[get_model] Fit autoencoder')

            loss = ae.model.history.history['loss'][-1]

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    learner_map = {
        'dt': DecisionTreeClassifier(),
        'rf': RandomForestClassifier(),
        'svm': SVC()
    }
    model = learner_map[learner]
    data.y_train = data.y_train.astype(float)

    data.x_train, data.x_test, data.y_train, data.y_test = \
        map(np.array, (data.x_train, data.x_test, data.y_train, data.y_test))
    data.y_train = data.y_train.squeeze()
    data.y_test = data.y_test.squeeze()

    if wfo:
        print('[get_model] Running wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running wfo')
    
    if smote:
        print('[get_model] Running smote')
        transform = Transform('smote')

        if n_class > 2 and len(data.y_train.shape) > 1:
            data.y_train = np.argmax(data.y_train, axis=1)

        transform.apply(data)

        if n_class > 2 and len(data.y_train.shape) == 1:
            data.y_train = to_categorical(data.y_train, n_class)

        print('[get_model] Finished running smote')
    
    return model, data


def split_data(filename: str, data: Data, n_classes: int) -> Data:
    if n_classes == 2:
        if filename == 'firefox.csv':
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == 'chromium.csv':
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0,
                                np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(
            data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 3, 1, np.where(
            data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 3, 1, np.where(
            data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4))))
    elif n_classes == 7:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(
            data.y_train < 6, 3, np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(
            data.y_test < 6, 3, np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6))))))
    else:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(data.y_train < 4, 3, np.where(
            data.y_train < 6, 4, np.where(data.y_train < 8, 5, np.where(data.y_train < 11, 6, np.where(data.y_train < 21, 7, 8))))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(data.y_test < 4, 3, np.where(
            data.y_test < 6, 4, np.where(data.y_test < 8, 5, np.where(data.y_test < 11, 6, np.where(data.y_test < 21, 7, 8))))))))

    if n_classes > 2:
        data.y_train = to_categorical(data.y_train, num_classes=n_classes, dtype=int)
        data.y_test = to_categorical(data.y_test, num_classes=n_classes, dtype=int)

    return data


def load_issue_lifetime_prediction_data(filename: str, n_classes: int) -> Data:
    df = pd.read_csv(f'./data/{filename}.csv')
    df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    _df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
    _df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
    _df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
    _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])
    _df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
    _df['s91'] = df['s9'].apply(lambda x: eval(x)[1])
    
    if filename == 'firefox':
        _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])
    
    x = _df.drop('y', axis=1)
    y = _df['y']
    
    data = Data(*train_test_split(x, y))
    data = split_data(filename, data, n_classes)
    return data
