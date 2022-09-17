from typing import Union, Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from tqdm import tqdm

from weelex import lexicon
from weelex import embeddings
from weelex.batchprocessing import batchprocessing
from weelex import ensemble
from weelex.trainer import TrainProcessor


class WEELexClassifier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                 test_size: float = None,
                 random_state: int = None,
                 n_jobs: int = 1,
                 progress_bar: bool = False,
                 **train_params) -> None:
        self._embeddings = self._make_embeddings(embeds)
        self._is_fit = False
        self._model = ensemble.FullEnsemble
        self._test_size = test_size
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._train_params = train_params
        self._use_progress_bar = progress_bar
        
        # setting up default objects
        self._main_keys = None
        self._support_keys = None
        self._models = None
        self._best_params = {}
        self._tuned_params = {}

    def set_params(self, **params):
        self._train_params = params
        return self

    def get_params(self, deep: bool = True) -> dict:
        return self._train_params

    def fit(self, X, y):
        # TODO: make fit method that can handle hyperparameter tuning
        # Concept:
        #    - Challenge: tuning of hyperparametes should be allowed for each category separately?
        pass

    def weelexfit(self,
                  lex: Union[lexicon.Lexicon, dict, str],
                  support_lex: Union[lexicon.Lexicon, dict, str] = None,
                  main_keys: Iterable[str] = None,
                  support_keys: Iterable[str] = None,
                  hp_tuning: bool = True,
                  n_iter: int = 150,
                  param_grid: dict = None,
                  fixed_params: dict = None,
                  n_best_params: int = 3,
                  progress_bar: bool = False) -> None:
        self._setup_trainprocessor(lex, support_lex, main_keys, support_keys)

        # loop over the categories:
        models = []
        for cat in self._set_progress_bar(self._main_keys):
            if hp_tuning:
                self._hyperparameter_tuning(cat=cat,
                                            n_iter=n_iter,
                                            param_grid=param_grid,
                                            fixed_params=fixed_params,
                                            n_best_params=n_best_params)
            model_params = self._make_train_params(cat,
                                                   self._train_params,
                                                   self._tuned_params,
                                                   fixed_params)
            model = self._model(cat, **model_params)
            model.fit(*self._trainprocessor.feed_cat_Xy(cat=cat, train=True))
            models.append(model)
        self._models = models
        self._is_fit = True

    def _setup_trainprocessor(self,
                              lex: Union[lexicon.Lexicon, dict, str],
                              support_lex: Union[lexicon.Lexicon, dict, str] = None,
                              main_keys: Iterable[str] = None,
                              support_keys: Iterable[str] = None,) -> None:
        self._trainprocessor = TrainProcessor(lex=lex,
                                              support_lex=support_lex,
                                              embeddings=self._embeddings,
                                              test_size=self._test_size,
                                              random_state=self._random_state)
        self._trainprocessor.make_train_test_data()
        self._main_keys = self._trainprocessor.main_keys
        self._support_keys = self._trainprocessor.support_keys

    def _set_progress_bar(self):
        if self._use_progress_bar:
            return tqdm
        else:
            return self._emptyfunc

    @staticmethod
    def _emptyfunc(array):
        return array

    @staticmethod
    def _make_train_params(cat: str, *param_sets: dict) -> dict:
        new_params = {}
        if param_sets is not None:
            for dct in param_sets:
                if dct is not None:
                    if cat in dct.keys():
                        new_params.update(dct[cat])
                    else:
                        new_params.update(dct)
        return new_params

    def _hyperparameter_tuning(self, 
                               cat, 
                               n_iter,
                               param_grid,
                               fixed_params,
                               n_best_params=3):
        if param_grid is None:
            raise ValueError('No parameter grid set for tuning.')

        search = RandomizedSearchCV(
                estimator=ensemble.AugmentedEnsemble(cat, **fixed_params)),
                param_distribution=param_grid,
                n_iter=n_iter,
                n_jobs=self._n_jobs,
                cv=5,
                random_state=self._random_state)
        search.fit(*self._trainprocessor.feed_cat_Xy(cat=cat, train=True))
        self._best_params.update({cat: search.best_n_params_})
        results = search.cv_results_
        result_params = self._get_best_params(results, n_best_params)
        self._tuned_params.update({cat: result_params})
        
    @staticmethod    
    def _get_best_params(results, n_best_params):
        """_summary_

        Args:
            results (_type_): _description_
        """
        result_params = []
        for x, y, z in zip(results['rank_test_score'], 
                           results['params'], 
                           results['mean_test_score']):
            if x <= n_best_params:
                print(y, z)
                result_params.append(y)
        return result_params

    def predict(self,
                X: pd.DataFrame,
                cutoff: float = 0.5,
                n_batches: int = None,
                checkpoint_path: str = None) -> pd.DataFrame:
        preds = self.predict_proba(X=X)
        return (preds >= cutoff).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: implement predict_proba() method that can handle hyperparam. t.
        #       (i.e. for predicting lexicon.Lexicon words)
        pass

    def weelexpredict_proba(self,
                            X: pd.DataFrame,
                            n_batches: int = None,
                            checkpoint_path: str = None) -> pd.DataFrame:
        kwargs = {'X': X,
                  'n_batches': n_batches,
                  'checkpoint_path': checkpoint_path}
        if self._use_progress_bar:
            result = weelexpredict_proba_pb(**kwargs)
        else:
            result = weelexpredict_proba_nopb(X)

    @batchprocessing.batch_predict
    def weelexpredict_proba_pb(self,
                            X: pd.DataFrame,
                            n_batches: int = None,
                            checkpoint_path: str = None) -> pd.DataFrame:
        return weelexpredict_proba_nopb(X)

    def weelexpredict_proba_nopb(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: implement predict method for final text prediction
        pass

    def weelexpredict(self,
                      X: pd.DataFrame,
                      cutoff: float = 0.5,
                      n_batches: int = None,
                      checkpoint_path: str = None) -> pd.DataFrame:
        preds = self.weelexpredict_proba(X=X,
                                         n_batches=n_batches,
                                         checkpoint_path=checkpoint_path)
        return (preds >= cutoff).astype(int)

    def save(self):
        # TODO: Implement save() method
        pass

    def load(self):
        # TODO: Implement load() method
        pass

    def __repr__(self, N_CHAR_MAX=700):
        return super().__repr__(N_CHAR_MAX)

    @staticmethod
    def _make_embeddings(embeds: Union[embeddings.Embeddings, dict]):
        if not isinstance(embeds, embeddings.Embeddings):
            my_embeds = embeddings.Embeddings(embeds)
        else:
            my_embeds = embeds
        return my_embeds

    def _get_full_vocab(self) -> list:
        v1 = self._lexicon.get_vocabulary()
        v2 = self._support_lexicon.get_vocabulary()
        return sorted(list(set([v1, v2])))

    def _filter_embeddings(self) -> None:
        vocab = self._get_full_vocab()
        self._embeddings.filter_terms(vocab)

    @property
    def main_keys(self) -> list:
        return self._main_keys

    @property
    def support_keys(self) -> list:
        return self._support_keys
