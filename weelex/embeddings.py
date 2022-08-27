from dataclasses import dataclass
from typing import Union, Tuple, ClassVar, Iterable

import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import FastText
from gensim.models import Word2Vec



@dataclass
class Embeddings:
    def __init__(self, inputs) -> None:
        # if isinstance(inputs, dict):
        #     # TODO: make sure that self._keys and self._vectors would not appear in fields
        #     self._keys: ClassVar, self._vectors: ClassVar = self._data_from_dct(inputs)
        self.isfiltered = False


    # TODO: how do I handle these well?
    def _load_facebook_vectors(self, path):
        wv = load_facebook_vectors(path)
        self.wv = wv


    def load_finetuned_fasttext(self, path):
        wv = FastText.load(path).wv
        self.wv = wv


    def load_finetuned_word2vec(self, path):
        wv = Word2Vec.load(path).wv
        self.wv = wv


    def filter_terms(self, terms: Union[list, np.ndarray, pd.DataFrame, tuple]) -> None:
        keys, vectors = self._data_from_dct(
            {term: self._vectors[term] for term in terms}
        )
        self._keys = keys
        self._vectors = vectors
        self.isfiltered = True


    def _data_from_dct(dct:dict) -> Tuple[list, np.array]:
        testvalue = dct.values()[0]
        vectors = np.zeros( (len(dict, len(testvalue))) )
        keys = np.array(dct.keys())
        for i, key in enumerate(keys):
            vectors[i] = dct[key]
        return keys, vectors


    def _get_val_from_key(self, key):
        if self.isfiltered:
            index = self._keys().tolist().index(key)
            result =  self._vectors[index, :]
        else:
            result = self.wv[key]
        return result


    def _get_val_from_key_vectorized(self, keys):
        # TODO: currently only working for filtered embeddings
        sorter = np.argsort(keys)
        values = sorter[np.searchsorted(self._vectors, keys, sorter=sorter)]
        return values


    def save(self, path: str) -> None:
        pass


    def load(self, path: str) -> None:
        pass


    def lookup(self, terms: Union[str, list, np.ndarray, pd.Series]):
        if isinstance(terms, str):
            values =  self._get_val_from_key(terms)
        else:
            if not isinstance(terms, np.ndarray):
                terms = np.array(terms)
            values = self._get_val_from_key_vectorized(terms)
        return values


    def __getitem__(self, key):
        return self.lookup(key)



def load_embeddings(path):
    if embedding_type=='fasttext':
        if fine_tuning_version!='original':
            if filtered:
                with open(file_path, 'rb') as f:
                    wv = pickle.load(f)
            else:
                wv = FastText.load(file_path).wv
        else:
            if filtered:
                with open(os.path.join(file_path), 'rb') as f:
                    wv = pickle.load(f)
            else:
                wv = load_facebook_vectors(fasttext_path)
    else:
        if fine_tuning_version!='original':
            if filtered:
                with open(file_path, 'rb') as f:
                    wv = pickle.load(f)
            else:
            wv = Word2Vec.load(file_path).wv
        else:
            raise ValueError('Fine tuned Word2Vec model not implemented yet')

    # testing if embeddings work as intended:
    _ = wv[test_word]

    return wv


def get_embedding_file_path(output_folder,
                            embedding_type,
                            fine_tuning_version,
                            tfidf_type,
                            filtered=False):
    if fine_tuning_version=='original':
        path = os.path.join(output_folder, '{}_{}'.format(embedding_type, fine_tuning_version))
    else:
        path = os.path.join(output_folder, 'finetuned_{}_{}'.format(embedding_type, fine_tuning_version))

    if filtered:
        path += f'{tfidf_type}_filtered.p'
    else:
        path += '.model'

    return path

