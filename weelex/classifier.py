from typing import Union, Iterable

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from weelex import lexicon
from weelex import embeddings
from weelex.batchprocessing import batchprocessing


class WEELexClassifier(BaseEstimator, TransformerMixin):
    # def __init__():
    #     pass

    def fit(self,
            embeds: Union[dict, embeddings.Embeddings],
            lex: Union[lexicon.Lexicon, dict, str],
            support_lex: Union[lexicon.Lexicon, dict, str]=None,
            main_keys: Iterable[str]=None,
            support_keys: Iterable[str]=None,
            ) -> None:
        self._embeddings = self._make_embeddings(embeds)
        self._handle_lexica(main_lex=lex,
                            support_lex=support_lex,
                            main_keys=main_keys,
                            support_keys=support_keys)
        # TODO: Implement fit() method
        pass


    def predict(self,
                X: pd.DataFrame,
                cutoff: float=0.5,
                n_batches: int=None,
                checkpoint_path: str=None) -> pd.DataFrame:
        preds = self.predict_proba(X=X,
                                   n_batches=n_batches,
                                   checkpoint_path=checkpoint_path)
        return (preds>=cutoff).astype(int)


    @batchprocessing.batch_predict
    def predict_proba(self,
                      X: pd.DataFrame,
                      n_batches: int=None,
                      checkpoint_path: str=None) -> pd.DataFrame:
        # TODO: implement predict_proba() method
        pass


    def save(self):
        # TODO: Implement save() method
        pass


    def load(self):
        # TODO: Implement load() method
        pass


    def __repr__(self, N_CHAR_MAX=700):
        return super().__repr__(N_CHAR_MAX)


    def _handle_lexica(self,
                       main_lex: Union[lexicon.Lexicon, dict, str],
                       support_lex: Union[lexicon.Lexicon, dict, str]=None,
                       main_keys: Iterable[str]=None,
                       support_keys: Iterable[str]=None) -> None:
        # create lexica as Lexicon Instances:
        _main_lex = self._make_lexicon(main_lex)
        if support_lex is not None and support_lex:
            _support_lex = self._make_lexicon(support_lex)

        # get keys:
        if main_keys and main_keys is not None:
            _main_keys = main_keys
        else:
            _main_keys = _main_lex.keys()

        if support_keys and support_keys is not None:
            _support_keys = support_keys
        elif support_lex and support_lex is not None:
            _support_keys = _support_lex.keys()
        else:
            _support_keys = []

        # Remove any overlapping keys:
        _support_keys = [x for x  in _support_keys if x not in _main_keys]

        # Merge lexica:
        _full_lex = self._merge_lexica([_main_lex, _support_lex])

        self._lexicon = _full_lex
        self._main_keys = _main_keys
        self._support_keys = _support_keys


    def _make_lexicon(self, lex: Union[lexicon.Lexicon, dict, str]) -> lexicon.Lexicon:
        """Create proper Lexicon instance from the passed input lexicon

        Args:
            lexicon (dict, str or pd.DataFrame): The key-value pairs to use
                for the lexicon. If str is passed, it should be the path to
                a csv (containing tabular data) or json (containig key-value pairs)
                file. If the file ends with ".json", it will attempt to read the
                file with the json module. Otherwise pd.read_csv is attempted.

        Returns:
            lexicon.Lexicon: lexicon.Lexicon instance
        """
        if not isinstance(lex, lexicon.Lexicon):
            my_lex = lexicon.Lexicon(lex)
        else:
            my_lex = lex
        return my_lex


    def _merge_lexica(self, lexica: Iterable[lexicon.Lexicon]) -> lexicon.Lexicon:
        base_lex = lexica[0]
        full_lex = base_lex.merge(lexica[1:])
        return full_lex


    def _make_embeddings(self, embeds: Union[embeddings.Embeddings, dict]):
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
