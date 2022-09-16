from typing import Union, Iterable

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from weelex import lexicon
from weelex import embeddings
from weelex.batchprocessing import batchprocessing
from weelex import ensemble
from weelex.trainer import TrainProcessor


class WEELexClassifier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                 test_size: float=None,
                 random_state=None,
                 **train_params) -> None:
        self._embeddings = self._make_embeddings(embeds)
        self._is_fit = False
        self._model = ensemble.FullEnsemble
        self._test_size = test_size
        self._random_state = random_state
        self._train_params = train_params
        self._main_keys = None
        self._support_keys = None

    def set_params(self, params: dict) -> None:
        self._train_params = params

    def get_params(self) -> dict:
        return self._train_params

    def fit(self, X, y):
        # TODO: make fit method that can handle hyperparameter tuning
        pass

    def weelexfit(self,
                  lex: Union[lexicon.Lexicon, dict, str],
                  support_lex: Union[lexicon.Lexicon, dict, str]=None,
                  main_keys: Iterable[str]=None,
                  support_keys: Iterable[str]=None,) -> None:
        self._trainprocessor = TrainProcessor(lex=lex,
                                              support_lex=support_lex,
                                              embeddings=self._embeddings,
                                              test_size=self._test_size,
                                              random_state=self._random_state)
        self._trainprocessor.make_train_test_data()
        self._main_keys = self._trainprocessor.main_keys
        self._support_keys = self._trainprocessor.support_keys

        # loop over the categories:
        models = []
        for cat in self._main_keys:
            model = self._model(cat, **self._train_params)
            model.fit(*self._trainprocessor.feed_cat_Xy(cat=cat, train=True))
            models.append(model)
        self._models = models
        self._is_fit = True

    def predict(self,
                X: pd.DataFrame,
                cutoff: float=0.5,
                n_batches: int=None,
                checkpoint_path: str=None) -> pd.DataFrame:
        preds = self.predict_proba(X=X)
        return (preds >= cutoff).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: implement predict_proba() method that can handle hyperparam. t. (i.e. for predicting lexicon.Lexicon words)
        pass

    @batchprocessing.batch_predict
    def weelexpredict_proba(self,
                            X: pd.DataFrame,
                            n_batches: int=None,
                            checkpoint_path: str=None) -> pd.DataFrame:
        # TODO: implement predict method for final text prediction
        pass

    def weelexpredict(self,
                      X: pd.DataFrame,
                      cutoff: float=0.5,
                      n_batches: int=None,
                      checkpoint_path: str=None) -> pd.DataFrame:
        preds = self.weelpredict_proba(X=X,
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
