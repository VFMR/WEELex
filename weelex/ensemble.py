import random
import pickle
import  glob
import os
from re import I
from typing import Tuple, Iterable, Union, List

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from gensim.models.callbacks import CallbackAny2Vec
# from scipy.sparse import vstack


def make_agg_sample(X: np.ndarray, n: int = 3) -> np.ndarray:
    """Draw n vectors randomly and return the linear
       combination of these with random weights.
       Method to ensure npdarraothe vectors add up to 1 is simplified
       by just drawing random integers and taking their relative size.

    Args:
        X (numpy.array): n-m Matrix with data to draw from.
        n (int): Number of vectors to aggregate.

    Returns:
        numpy.array: Vector with linear combination of shape (m,).
            i.e. is only one observation.
    """
    input_shape = X.shape
    error = True

    if isinstance(X, pd.DataFrame):
        X = np.array(X)

    # draw random integers to compute weights:
    while error:  # simple way to repeat drawing if all n weights are 0
        weights = [random.randint(0, 10) for _ in range(n)]
        if sum(weights) != 0:
            error = False

    # compute weights by taking relative size
    weights = np.array([x/sum(weights) for x in weights])

    # draw n random vector indices
    # random_vect_ix = [random.randint(0, len(X)-1) for _ in range(n)]
    random_vect_ix = [random.randint(0, input_shape[0]-1) for _ in range(n)]

    # combine the random vectors into a matrix
    vects = np.zeros((input_shape[1], n))
    for i in range(n):
        vects[:, i] = X[random_vect_ix[i], :]

    # compute linear combination:
    new_vector = vects @ weights
    return new_vector


class BaseEnsemble:
    def __init__(self, progress_bar=False):
        self.progress_bar_func = self._get_progress_bar_func(progress_bar)

    def _get_progress_bar_func(self, progress_bar):
        if progress_bar:
            progress_bar_func = self._use_progress_bar
        else:
            progress_bar_func = self._no_progress_bar
        return progress_bar_func

    def _no_progress_bar(self, array):
        return array

    def _use_progress_bar(self, array):
        return tqdm(array)


class AugmentedEnsemble(BaseEnsemble, BaseEstimator):
    """Class to train an ensemble of models.
    For each model, a random selection of input vectors
    is aggregated as a linear combination to "augment" the data.

    Args:
        BaseEstimator: Sklearn base class for estimators.
    """
    def __init__(self,
                 category,
                 categories,
                 outside_categories,
                 n_models=10,
                 n_samples_multiplier=5,
                 n_samples_multiplier_outside=5,
                 input_shape=300,
                 modeltype='svm',
                 pca=False,
                 svc_c=1.0,
                 svc_kernel='rbf',
                 # mlp_layers=[30, 10],
                 # mlp_lr=0.01,
                 # mlp_regularization=False,
                 progress_bar=False,
                 n_vectors_agg_training=3,
                 run_fast=False,
                 random_state=None,
                 **modelkwargs):
        """Initialization method

        Args:
            category (str): The category to be predicted vs. the rest
            n_models (int, optional): Number of models to for each category.
                Models differ only in the vectors that are randomly drawn for
                the observations of class 0. Defaults to N_MODELS.
            n_samples_multiplier (int, optional): Multiply with the number of
                words of the respective category to get the number of "inside"
                observations.
                Defaults to N_SAMPLES_MULTIPLIER.
            n_samples_multiplier_outside (int, optional): Multiplied with the
                number of words of the respective category, multiplied with
                n_samples_multiplier.
                I.e. we want n-times as many outside vectors as inside vectors.
                Defaults to N_SAMPLES_MULTIPLIER_OUTSIDE.
            input_shape (int, optional): Dimensionality of embedding vectors.
                Defaults to 300.
            modeltype (str, optional): Type of ml model to use. {'svm', 'mlp'}.
                Defaults to 'svm'.
            pca (bool, optional): whether or not to use Principal component
                analysis.
                Defaults to False.
            svc_c (float, optional): Hyperparameter C of SVC. Only used when
                modeltype == 'svm'.
                Defaults to 1.0.
            svc_kernel (str, optional): Hyperparameter kernel function of SVC.
                Only used when modeltype == 'svm'. Defaults to 'rbf'.
            mlp_layers (list, optional): Layers of MLPClassifier. Only use
                when modeltype=='mlp'.
                Defaults to [30,10].
            mlp_lr (float, optional): Learning Rate of MLPClassifier. Only
                used when modeltype=='mlp'.
                Defaults to 0.01.
            mlp_regularization (bool, optional): Regularization for
                MLPClassifier. Only used when modeltype=='mlp'.
                Defaults to False.
            progress_bar (bool, optional): whether or not to use a tqdm
                progress bar.
                Defaults to False.
            n_vectors_agg_training (int, optional): Number of vectors to
                aggregate for training.
                Defaults to N_VECTORS_AGG_TRAINING.
            categories (list, optional): List of categories to train.
                Defaults to LIWC_CATEGORIES_ALL.
            outside_categories (list, optional): List of categories where no
                model is fit
                but whose words are still used as outside vectors for training.
                Defaults to LIWC_CATEGORIES_OUTSIDE.
            run_fast (bool, optional): If True, do not compute probabilities of
                each SVM model.
                If False, compute probability of each model via 5 fold CV.
        """
        self.category = category
        self.n_models = n_models
        self.modeltype = modeltype
        self.input_shape = input_shape
        self.n_samples_multiplier = n_samples_multiplier
        self.n_samples_multiplier_outside = n_samples_multiplier_outside
        self.pca = pca
        self.svc_c = svc_c
        self.svc_kernel = svc_kernel
        # self.mlp_layers = mlp_layers
        # self.mlp_lr = mlp_lr
        # self.mlp_regularization = mlp_regularization
        self.modelkwargs = modelkwargs
        self.progress_bar = progress_bar
        self.n_vectors_agg_training = n_vectors_agg_training
        self.categories = categories
        self.outside_categories = outside_categories

        # combine inside and outside categories:
        self.outside_categories_all = categories + outside_categories

        if progress_bar:
            self.progress_func = self._use_progress_bar
        else:
            self.progress_func = self._no_progress_bar

        self.run_fast = run_fast

        self.random_state = random_state
        if random_state and random_state is not None:
            random.seed(random_state)

    @staticmethod
    def _get_targets(y: np.ndarray) -> np.ndarray:
        if len(y.shape) > 1:
            targets = y[:, 0]
        else:
            targets = y
        return targets

    @staticmethod
    def _get_drawclass_data(array: Union[list, pd.DataFrame, np.ndarray],
                            targets: np.ndarray,
                            classvalue: int) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            array_np = np.array(array)
        else:
            array_np = array
        return array_np[targets == classvalue]

    @staticmethod
    def _random_category(array: List[str], cat: str) -> str:
        """Randomly selects a category from a list

        Example:
            >>> oc = ['Cars', 'Politics']
            >>> ic = ['Food', 'Space']
            >>> cc = 'Politics'
            >>> x = AugmentedEnsemble(category=cc, categories=ic, outside_categories=oc)
            >>> rc = x._random_category(oc, cc)
            >>> rc in ['Food', 'Cars', 'Space']
            True

        Args:
            array (List[str]): Array to choose from.
            cat (str): Category that shall not be picked

        Returns:
            str: selected item
        """
        random_cat = random.choice(
                [x for x in array if x != cat])
        return random_cat

    def _getkeeps(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: np.ndarray,
                  classvalue: int,
                  ) -> List[bool]:
        """Draw a random category for the observation:
           avoid drawing and combining words from different categories.
           Their linear combination could be anywhere.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): _description_
            y (Union[pd.DataFrame, np.ndarray]): _description_
            classvalue (int): _description_

        Returns:
            list: List of booleans
        """
        if classvalue == 1:
            # the "inside" class is already coming from a single category
            # just create an indicator that all vectors can be used
            keep = [True]*len(X)  # keep all
            random_cat = None
        else:
            # The "outside" class is made up of multiple LIWC categories.
            # randomly select a category and make an array that
            # flags vectors that belong to said category:
            random_cat = self._random_category(array=self.outside_categories_all,
                                               cat=self.category)
            if not isinstance(y, np.ndarray):
                y_np = np.array(y)
            else:
                y_np = y
            keep = list(y_np[:, 1] == random_cat)
        return keep

    def draw_random_samples_classwise(self,
                                      X: pd.DataFrame,
                                      y: pd.DataFrame,
                                      classvalue: int,
                                      n_samples_multiplier: int = 5,
                                      n_samples: int = None,
                                      include_all_original: bool = True
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine word vectors by randomly selecting terms
        of a specific category and making a linear combination of these

        Args:
            X (pandas.DataFrame): The matrix of word vectors
            y (pandas.DataFrame): Contains the classes of each term
            classvalue (int): Value of current class to consider. 0 or 1
            n_samples_multiplier (int): Value to multiply number of original
                words of the category with to get the number of observations
                that shall be drawn.
            n_samples (int, optional): Number of samples. Defaults to None.
            include_all_original (bool, optional): Whether the
                original, i.e. non-aggregated, word vectors shall be
                included in the sample. Defaults to True.

        Returns:
            numpy.ndarray, numpy.ndarray: The X and y arrays that can be used
                for input in Machine Learning models.
        """
        # y might be a matrix -> reduce this to a vector containing the binary
        # class value
        y_orig = y
        y = np.array(y)
        targets = self._get_targets(y)

        # get arrays with the data of the class currently drawn
        n_original_samples = len(X[targets == 1])
        X = self._get_drawclass_data(X, targets, classvalue)
        y = self._get_drawclass_data(y, targets, classvalue)

        # get counts for the number of original terms and
        # the number of observations that shall be drawn.
        if n_samples_multiplier:
            n_samples = n_original_samples * n_samples_multiplier

        # generate samples:
        new_X = []
        append = new_X.append

        # draw individual observations (word vectors)
        for _ in range(n_samples):

            # Draw a random category for the observation:
            # avoid drawing and combining words from different categories.
            # Their linear combination could be anywhere.
            keep = self._getkeeps(X, y, classvalue)

            # aggregate random vectors from the randomly drawn category:
            append(make_agg_sample(X[keep], n=self.n_vectors_agg_training))

        # the target vector (the value to predict)
        # is just 1 or 0, depending on whether we draw outside
        # or inside vectors, for all observations
        new_y = [classvalue for _ in range(n_samples)]

        if include_all_original:
            X = np.array(X)

            if classvalue == 0:
                # for "outside" vectors, using *all* terms would be excessive
                # hence, randomly draw as many terms as are included in the
                # inside category
                random_ix = [random.randint(0, len(X)-1) for _ in range(n_original_samples)]
                new_X = np.concatenate([
                    np.array(new_X),  # the aggregated vectors
                    np.array([X[ix, :] for ix in random_ix])  # non-agg vectors
                ], axis=0)
            else:
                new_X = np.concatenate([
                    np.array(new_X),  # the aggregated vectors
                    X  # non-aggregated vectors
                    ], axis=0)

            new_y = new_y + [classvalue]*n_original_samples

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # just for random shuffling
        new_X, _, new_y, __ = train_test_split(new_X, new_y, test_size=1)
        return new_X, new_y

    def _setup_pca(self):
        if self.pca is not False:
            if isinstance(self.pca, float) and self.pca < 1:
                n_components = int(self.input_shape * self.pca)
            else:
                n_components = self.pca
        pca = PCA(n_components=n_components)
        return pca

    def _svm_model(self):
        if self.pca is not False:
            steps = [
                ('scaler', StandardScaler()),
                ('pca', self._setup_pca()),
                ('svm', SVC(C=self.svc_c, class_weight='balanced', kernel='rbf'))
            ]
        else:
            steps = [
                ('scaler', StandardScaler()),
                ('svm', SVC(C=self.svc_c, class_weight='balanced', kernel='rbf'))
            ]

        model = Pipeline(steps)
        return model

    def _build_models(self):
        """Set up a list of n_models untrained ml models

        Returns:
            list: List of models
        """
        if self.modeltype == 'mlp':
            models = [self._mlp_model() for _ in range(self.n_models)]
        elif self.modeltype == 'svm':
            models = [self._svm_model() for _ in range(self.n_models)]
        return models

    # def _no_progress_bar(self, array):
    #     """Returns the array that is input.
    #     I.e. this method does basically nothing and serves
    #     just as a placeholder or replacement for the progress
    #     bar method if no progress bar shall be shown.

    #     Args:
    #         array (numpy.array): Array to loop over

    #     Returns:
    #         numpy.array: The array that is passed with no change to it.
    #     """
    #     return array

    # def _use_progress_bar(self, array):
    #     return tqdm(array)

    def fit(self, X, y):
        """Method to train the ml models via passed X and y arrays.
        X is a matrix with the embedding vectors for each term, y is
        a binary that tells whether the term belongs to the "inside" (i.e.
        is part of the category that shall be predicted) or the "outside"
        (i.e. is from a category different from the one to be predicted).

        A sample of observations, in the form of linear combinations from
        random word embeddings of a category, is drawn and used for training.

        Args:
            X (numpy.array or pandas.DataFrame): Matrix of word embeddings
            y (numpy.array or pandas.DataFrame): target values
        """
        # reinstantiate models to reset weights...
        self.models = self._build_models()

        modelkwargs = self.modelkwargs
        ns_m = self.n_samples_multiplier
        ns_m_out = self.n_samples_multiplier_outside

        # "inside" data, i.e. the vectors that belong to the data to predict
        X1, y1 = self.draw_random_samples_classwise(X, y,
                                                    classvalue=1,
                                                    n_samples_multiplier=ns_m,
                                                    include_all_original=True)

        trained_models = []
        append = trained_models.append

        # multiple models are combined in an ensemble.
        # The outside vectors are drawn separately for each model
        # This is supposed to allow for model different cases
        for model in self.progress_func(self.models):
            X0, y0 = self.draw_random_samples_classwise(
                X,
                y,
                classvalue=0,  # i.e. the categories that are not to predict
                n_samples_multiplier=ns_m*ns_m_out,
                include_all_original=True)

            # combine inside and outside vectors:
            X_train = np.concatenate([np.array(X1), np.array(X0)], axis=0)
            y_train = np.concatenate([np.array(y1), np.array(y0)], axis=0)
            # random shuffling:
            X_train, _, y_train, __ = train_test_split(X_train,
                                                       y_train,
                                                       test_size=2)

            if self.pca and isinstance(self.pca, int):
                n_components = min(X_train.shape[0], self.pca)-1
                # modelkwargs.update({'pca__n_components': n_components})
                model.set_params(pca__n_components=n_components)

            if self.run_fast:
                model.set_params(svm__probability=False)
            else:
                model.set_params(svm__probability=True)

            model.fit(X_train, y_train, **modelkwargs)
            append(model)
        self.models = trained_models

    def predict_proba(self, X):
        """Method to predict "probabilities" for each observation.
        Mean of probabilites for individual models.

        Args:
            X (numpy.array or pandas.DataFrame): Matrix to predict

        Returns:
            numpy.array: Prediction vector containing the probabilities
        """
        allpreds = np.zeros((len(X), len(self.models)))
        i = 0
        for model in self.progress_func(self.models):
            if self.modeltype == 'mlp':
                preds = model.predict(X, verbose=0)
            else:
                if self.run_fast:
                    preds = model.predict(X)
                else:
                    preds = model.predict_proba(X)
                    preds = preds[:, 1]  # only proba for class 1
            allpreds[:, i] = preds.reshape(len(X))
            i += 1

        final_pred = np.mean(allpreds, axis=1)
        return final_pred

    def predict(self, X, cutoff=0.5):
        """Predict binary class of input matrix.
        Uses .predict_proba method.

        Args:
            X (array-like): Input matrix
            cutoff (float, optional): Cutoff value for binary prediction.
                Defaults to 0.5.

        Returns:
            numpy.array: Binary prediction
        """
        preds = self.predict_proba(X)
        return (preds > cutoff)

    def score(self, X, y, score_func=f1_score):
        """ Scoring function. Used for Parameter tuning

        Args:
            X (array-like): Input matrix
            y (array-like): target vector
            score_func (callable, optional): Function for scoring.
                Defaults to sklearn.metrics.f1_score.

        Returns:
            float: Score
        """
        if len(y.shape) > 1:
            y = y.iloc[:, 0].astype(int)
        preds = self.predict(X, cutoff=0.5)
        return score_func(y, preds)

    def load(self, filename):
        """Load (hyper-)parameters and models that were saved using
        the .save method.

        Args:
            filename (str): Name of saved file
        """
        # TODO: change to loading json
        with open(filename, 'rb') as f:
            parameters = pickle.load(f)

        self.category = parameters['category']
        self.modeltype = parameters['modeltype']
        self.n_models = parameters['n_models']
        self.n_samples_multiplier = parameters['n_samples_multiplier']
        self.input_shape = parameters['input_shape']
        self.svc_c = parameters['svc_c']
        self.svc_kernel = parameters['svc_kernel']
        # self.mlp_layers = parameters['mlp_layers']
        # self.mlp_lr = parameters['mlp_lr']
        # self.mlp_regularization = parameters['mlp_regularization']
        self.pca = parameters['pca']
        self.modelkwargs = parameters['modelkwargs']

        models = []
        append = models.append
        if '.p' in filename:
            filename = filename[:-2]

        for i in range(self.n_models):
            if self.modeltype == 'mlp':
                # append(keras_load(filename+'_model{}.h5'.format(i)))
                pass
            else:
                # TODO: use builtin .save method
                with open(filename+'_model{}.p'.format(i), 'rb') as f:
                    append(pickle.load(f))
        self.models = models

    def save(self, filename):
        """Save (hyper-)parameters and models to disk.
        Use this to use a fitted model later on for prediction.

        Args:
            filename (str): Path and name of save file.
        """
        # TODO: Change to saving json
        parameters = {
            'category': self.category,
            'modeltype': self.modeltype,
            'n_models': self.n_models,
            'n_samples_multiplier': self.n_samples_multiplier,
            'input_shape': self.input_shape,
            'svc_c': self.svc_c,
            'svc_kernel': self.svc_kernel,
            # 'mlp_layers': self.mlp_layers,
            # 'mlp_lr': self.mlp_lr,
            # 'mlp_regularization': self.mlp_regularization,
            'pca': self.pca,
            'modelkwargs': self.modelkwargs
        }

        with open(filename+'.p', 'wb') as f:
            pickle.dump(parameters, f)

        for i in range(self.n_models):
            if self.modeltype == 'mlp':
                self.models[i].save(filename+'_model{}.h5'.format(i))
            else:
                # use builtin .save method
                with open(filename+'_model{}.p'.format(i), 'wb') as f:
                    pickle.dump(self.models[i], f)



class FullEnsemble(BaseEnsemble):
    def __init__(self,
                 category,
                 categories,
                 outside_categories,
                 param_set=None,
                 progress_bar=False,
                 model=AugmentedEnsemble,
                 **modelparams):
        super().__init__(progress_bar)
        self.category = category
        self._categories = categories
        self._outside_categories = outside_categories
        self.param_set = param_set
        self.model = model
        self.fixedparams = modelparams

    def _build_models(self, param_set):
        full_param_set = []
        if param_set is None:
            param_set = []

        if len(param_set) > 0:
            for params in param_set:
                full_param_set.append({**self.fixedparams, **params})
            print('    Sets of parameters:')
            for i, x in enumerate(full_param_set):
                print(f'    {i}: {x}')

            models = [self.model(self.category,
                                 self._categories,
                                 self._outside_categories,
                                 **full_param_set[i]
                                 ) for i in range(len(param_set))]
        else:
            models = [self.model(self.category,
                                 self._categories,
                                 self._outside_categories,
                                 **self.fixedparams)]
        return models

    def fit(self, X, y):
        models = self._build_models(self.param_set)
        trained_models = []
        for model in self.progress_bar_func(models):
            model.fit(X, y)
            trained_models.append(model)
        self.trained_models = trained_models
        return self

    def predict(self, X, cutoff=0.5):
        return (self.predict_proba(X) > cutoff).astype(int)

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], len(self.trained_models)))
        for i, model in enumerate(self.trained_models):
            predictions[:, i] = model.predict_proba(X)
        return np.mean(predictions, axis=1)

    def save(self, path):
        for i, model in enumerate(self.trained_models):
            model.save(os.path.join(path,
                                    f'liwc_ensemble_{self.category}_{i}'))

    def load(self, path):
        # search for all model files in the path-folder
        mnames = glob.glob(os.path.join(path,
                                        f'liwc_ensemble_{self.category}*'))
        mnames = [
            x for x in mnames if 'model' not in x.replace('liwc_models', '')
                ]
        trained_models = []
        for name in mnames:
            model = self.model(self.category, **self.fixedparams)
            model.load(name)
            trained_models.append(model)
        self.trained_models = trained_models


def ensemble_cross_val_score(model, X, y, cv=5):
    X, _, y, _ = train_test_split(X, y, test_size=1)
    X_splits = np.array_split(X, cv)
    y_splits = np.array_split(y, cv)

    scores = []
    for i in range(len(X_splits)):
        X_train = pd.DataFrame(np.concatenate(
            [x for j, x in enumerate(X_splits) if j != i]
            ))
        y_train = pd.DataFrame(np.concatenate(
            [x for j, x in enumerate(y_splits) if j != i]
            ))
        X_test = pd.DataFrame(X_splits[i])
        y_test = pd.DataFrame(y_splits[i])
        print(y_test.iloc[:, 0].value_counts())
        print(X_train)

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    return scores
