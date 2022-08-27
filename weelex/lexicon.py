from dataclasses import dataclass
from typing import Union, Iterable
import json

import pandas as pd
import numpy as np

import weelex


@dataclass
class Lexicon:
    # TODO: Allow for train_test_split to work
    def __init__(self, dictionary: Union[dict, str, pd.DataFrame]):
        self._dictionary_df = self._build_dictionary(dictionary)
        self.keys = self._dictionary_df.columns

    def _build_dictionary(self,
                          dictionary: Union[dict, str, pd.DataFrame]
                          ) -> pd.DataFrame:
        """Processes the input lexicon and returns it in a pandas DataFrame.

        Args:
            dictionary (dict, str or pd.DataFrame): The key-value pairs to use
                for the lexicon. If str is passed, it should be the path to
                a csv (containing tabular data) or json (containig key-value pairs)
                file. If the file ends with ".json", it will attempt to read the
                file with the json module. Otherwise pd.read_csv is attempted.

        Returns:
            pd.DataFrame: lexicon matrix
        """
        if isinstance(dictionary, str):
            if dictionary.endswith('.json'):
                with open(dictionary, 'r') as f:
                    my_dct = json.loads(f.read())
                    dct_df = pd.DataFrame(dict_padding(my_dct))
            else:
                dct_df = pd.read_csv(dictionary)
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
        old_keys = old_dct.keys
        new_keys = lex.keys
        update_keys = [x for x in new_keys if x in old_keys]
        append_keys = [x for x in new_keys if x not in old_keys]
        new_dct = pd.concat([old_dct, lex.loc[:, append_keys]])
        # TODO: allow for merge of existing keys


    def _append_values(self, lex: 'Lexicon', key: str, values: pd.Series) -> pd.Series:
        maxlen = lex._dictionary_df.shape[0]
        collen = len(lex._dictionary_df[~lex._dictionary_df[key].isna()])
        # TODO: Implement rest of method _append_values()


    def get_vocabulary(self) -> list:
        vocab = []
        for col in self._dictionary_df:
            vocab += list(self._dictionary_df[col])
        return sorted(list(set(vocab)))


    def to_dict(self) -> dict:
        """Return the lexicon in dictionary format.

        Returns:
            dict: dict with the lexicon key-value pairs
        """
        return self._dictionary_df.to_dict()


    def save(self, path: str) -> None:
        """Save lexicon to disk

        Args:
            path (str): Output file path
        """
        # TODO: implement save method
        pass


    def load(self, path: str) -> None:
        """Load a previously saved Lexicon instance

        Args:
            path (str): Path of saved Lexicon instance
        """
        # TODO: implement load method
        pass


def dict_padding(dictionary: dict, filler=np.nan) -> dict:
    """Padding of a dictionary where the values are lists such that these lists
    have the same length.
    Example: {'A': [1,2,3], 'B': [1,2]} -> {'A': [1,2,3], 'B': [1,2,np.nan]}

    Args:
        dictionary (dict): Dictionary where the values are lists,
            e.g. {'A': [1,2,3], 'B': [1,2]}
        filler: Value to pad with. Defaults to np.nan

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
