from dataclasses import dataclass
from typing import Union, Iterable
import json

import pandas as pd
import numpy as np


@dataclass
class Lexicon:
    # TODO: Allow for train_test_split to work
    def __init__(self,
                 dictionary: Union[dict, str, pd.DataFrame],
                 sep: str=None,
                 encoding: str=None):
        self._dictionary_df = self._build_dictionary(dictionary, sep=sep, encoding=encoding)


    def __getitem__(self, key) -> pd.Series:
        return self._dictionary_df[key]


    def __setitem__(self, key, value) -> None:
        self._dictionrary_df[key] = value


    def _build_dictionary(self,
                          dictionary: Union[dict, str, pd.DataFrame],
                          sep: str=None,
                          encoding: str=None
                          ) -> pd.DataFrame:
        """Processes the input lexicon and returns it in a pandas DataFrame.

        Args:
            dictionary (dict, str or pd.DataFrame): The key-value pairs to use
                for the lexicon. If str is passed, it should be the path to
                a csv (containing tabular data) or json (containig key-value pairs)
                file. If the file ends with ".json", it will attempt to read the
                file with the json module. Otherwise pd.read_csv is attempted.
            sep (str, optional): separator character when reading csv file

        Example:
            >>> my_dct = {'A': ['a', 'b'], 'B': ['c', 'd']}
            >>> l = Lexicon(my_dct)
            >>> l._build_dictionary(my_dct)
                 A    B
            0    a    c
            1    b    d

        Returns:
            pd.DataFrame: lexicon matrix
        """
        if isinstance(dictionary, str):
            if dictionary.endswith('.json'):
                with open(dictionary, 'r') as f:
                    my_dct = json.loads(f.read())
                    dct_df = pd.DataFrame(dict_padding(my_dct))
            else:
                dct_df = pd.read_csv(dictionary, sep=sep, encoding=encoding)
        elif isinstance(dictionary, pd.DataFrame):
            dct_df = dictionary
        elif isinstance(dictionary, dict):
            dct_df = pd.DataFrame(dict_padding(dictionary))

        return dct_df


    def merge(self, lexica: Union['Lexicon', Iterable['Lexicon']]) -> None:
        if isinstance(lexica, Lexicon):
            self._merge_one(lexica)
        else:
            for lex in lexica:
                self._merge_one(lex)


    def _merge_one(self, lex: 'Lexicon') -> None:
        old_dct = self._dictionary_df.copy()
        old_keys = old_dct.keys()
        new_keys = lex.keys
        update_keys = [x for x in new_keys if x in old_keys]
        append_keys = [x for x in new_keys if x not in old_keys]
        new_dct = pd.concat([old_dct, lex._dictionary_df.loc[:, append_keys]],
                            # ignore_index=True,
                            axis=1)
        self._dictionary_df = new_dct
        # TODO: allow for merge of existing keys

    def _append_values(self, lex: 'Lexicon', key: str, values: pd.Series) -> pd.Series:
        maxlen = lex._dictionary_df.shape[0]
        collen = len(lex._dictionary_df[~lex._dictionary_df[key].isna()])
        # TODO: Implement rest of method _append_values()

    @property
    def keys(self):
        return list(self._dictionary_df.columns)

    def get_vocabulary(self) -> list:
        """Returns a sorted list of all the lexicon categories

        Example:
            >>> my_dct = {'Food': ['Bread', 'Salad'], 'Animals': ['Dog', 'Cat']}
            >>> l = Lexicon(my_dct)
            >>> l.get_vocabulary()
            ['Bread', 'Cat', 'Dog', 'Salad']

        Returns:
            list: Sorted array of categories
        """
        vocab = []
        for col in self._dictionary_df:
            vocab += list(self._dictionary_df[col])
        vocab = [x for x in vocab if isinstance(x, str)]  # remove np.nans
        return sorted(list(set(vocab)))

    @property
    def vocabulary(self) -> list:
        return self.get_vocabulary()

    def to_dict(self) -> dict:
        """Return the lexicon in dictionary format.

        Example:
            >>> my_dct = {'A': ['a', 'b'], 'B': ['c', 'd']}
            >>> l = Lexicon(my_dct)
            >>> l.to_dict()
            {'A': ['a', 'b'], 'B': ['c', 'd']}

        Returns:
            dict: dict with the lexicon key-value pairs
        """
        out_dict = {
            key: list(
                self._dictionary_df[key]
                ) for key in self._dictionary_df.columns}
        return out_dict


    def save(self, path: str) -> None:
        """Save lexicon to disk

        Args:
            path (str): Output file path
        """
        # TODO: implement save method
        pass


    @classmethod
    def load(cls, path: str):
        """Load a previously saved Lexicon instance

        Args:
            path (str): Path of saved Lexicon instance
        """
        # TODO: implement load method
        # return cls()


def dict_padding(dictionary: dict, filler=np.nan) -> dict:
    """Padding of a dictionary where the values are lists such that these lists
    have the same length.

    Args:
        dictionary (dict): Dictionary where the values are lists,
            e.g. {'A': [1,2,3], 'B': [1,2]}
        filler: Value to pad with. Defaults to np.nan

    Example:
        >>> dict_padding({'A': [1,2,3], 'B': [1,2]})
        {'A': [1, 2, 3], 'B': [1, 2, nan]}

    Returns:
        dict: Padded dictionary
    """
    lenghts = [len(dictionary[x]) for x in dictionary.keys()]
    maxlen = max(lenghts)
    padded_dict = {}
    for key, value in dictionary.items():
        new_list = list_padding(lst=value, maxlen=maxlen, filler=filler)
        padded_dict.update({key: new_list})
    return padded_dict


def list_padding(lst: list, maxlen: int, filler=np.nan) -> list:
    """Appends a filler value to a list such that the list has the preferred
    length, or cut values at the end

    Args:
        lst (list): List to fill
        maxlen (int): Desired length
        filler (optional): Value to fill list with. Defaults to np.nan.

    Example:
        >>> list_padding(lst=[1,2], maxlen=4, filler=0)
        [1, 2, 0, 0]
        >>> list_padding(lst=[1,2,3], maxlen=2, filler=0)
        [1, 2]
        >>> list_padding(lst=[1,2,3], maxlen=5, filler=np.nan)
        [1, 2, 3, nan, nan]

    Returns:
        list: padded list
    """
    len_diff = maxlen - len(lst)
    new_lst = lst.copy()
    if len_diff > 0:
        for _ in range(len_diff):
            new_lst.append(filler)
    if len_diff < 0:
        new_lst = new_lst[0:maxlen]
    return new_lst


def merge_lexica(lexica: Iterable[Lexicon]) -> Lexicon:
    lex1 = lexica[0]
    if len(lexica) > 1:
        for lex in lexica[1:]:
            lex1.merge(lex)
    return lex1
