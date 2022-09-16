from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from weelex import lexicon
from weelex import embeddings


class TrainProcessor:
    def __init__(self,
                 lex: lexicon.Lexicon,
                 support_lex: lexicon.Lexicon=None,
                 embeddings: embeddings.Embeddings=None,
                 split_train_test:bool=False,
                 test_size:Union[float, int]=0.2,
                 random_state:int=None):
        self.lex = lex
        self.support_lex = support_lex
        self.embeddings = embeddings
        self.split_train_test = split_train_test
        self.test_size = test_size
        self.random_state = random_state

    def _handle_outliers(self):
        # TODO: make method to remove outliers (usually 2 character abbrevs.)
        pass

    @staticmethod
    def _prepare_input_data(lex: lexicon.Lexicon):
        for cat in lex.keys:
            lex[cat] = lex._clean_strings(lex[cat])

        lex.embed()
    
    def _get_embedding_df(self, lex: lexicon.Lexicon) -> pd.DataFrame:
        if lex.embeddings is None or lex.embeddings.shape[1] != len(lex.keys):
            lex.embed(self.embeddings)
        dfs = []
        for i, cat in enumerate(lex.keys):
            # for j, word in enumerate(self[cat]):
            df = pd.DataFrame(lex.embeddings[:, i])
            # get nomissings:
            catw = lex[cat]
            nomiss = lex._nonmissarray(catw)

            df = df.iloc[:len(nomiss), :]
            df.index = cat+':'+nomiss
            df.index = df.index.str.replace('*', '', regex=False)
            dfs.append(df)
        df_full = pd.concat(dfs)
        return df_full

    @staticmethod
    def _make_id_term_mapping(embedding_df):
        """
        Example:
            >>> embedding_df = pd.DataFrame(np.zeros((3,10)))
            >>> embedding_df.index = ['Cars:Vehicle', 'Politics:Politician', 'Food:Bread']
            >>> x = TrainProcessor(lex=None)
            >>> x._make_id_term_mapping(embedding_df)
              categories       terms
            0       Cars     Vehicle
            1   Politics  Politician
            2       Food       Bread
        """
        # terms = embedding_df.index.str.replace(r'[A-Za-z0-9]*\:', '',
        #                                        regex=True)
        # term2cat = pd.DataFrame(terms, columns=['terms'])
        # categories = df_full.index.str.extract(r'([A-Za-z0-9]*)\:')
        # term2cat['categories'] = categories
        term2cat = embedding_df.index.str.split(':', expand=True)\
                                         .to_frame(index=False)
        term2cat.columns = ['categories', 'terms']
        return term2cat

    def _prepare_inputs(self):
        cats = self.lex.keys
        if self.support_lex is not None:
            outside_cats = self.support_lex.keys
        else:
            outside_cats = []
        lex_full = self.lex.merge(self.support_lex, inplace=False)
        embedding_df = self._get_embedding_df(lex_full)
        term2cat = self._make_id_term_mapping(embedding_df)
        self.cats = cats
        self.outside_cats = outside_cats
        self.lex_full = lex_full
        self.embedding_df = embedding_df
        self.term2cat = term2cat

    def _make_y(self):
        y_classes = {}
        terms_classes = {}

        for cat in self.cats:
            # separate terms from the current category and other categories
            cat_terms = self.term2cat[self.term2cat['categories']==cat].loc[:, ['terms']]
            cat_terms[cat] = True

            # "other" terms: words that do not belong to current category
            # but are nonetheless in our list of categories to consider.
            other_terms = self.term2cat[
                (self.term2cat['categories']!=cat) & (self.term2cat['categories'].isin(self.cats+self.outside_cats))
            ].loc[:, ['terms', 'categories']]

            # handle words that appear in multiple categories: remove from other category
            other_terms = other_terms[~other_terms['terms'].isin(list(cat_terms['terms']))]
            other_terms[cat] = False

            y = pd.concat([cat_terms, other_terms], ignore_index=False, axis=0)
            terms_classes.update({cat: y['terms']})
            del y['terms']
            y_classes.update({cat: y})

        return y_classes

    def _make_X(self, y_classes: dict):
        X_full = self.embedding_df.copy()
        X_full.index = np.arange(len(X_full))
        X_classes = {}
        for cat in self.cats:
            X_class_this = X_full.loc[y_classes[cat].index.values, :]
            X_classes.update({cat: X_class_this})
        return X_classes

    def _make_data(self):
        self._prepare_inputs()
        y_classes = self._make_y()
        x_classes = self._make_X(y_classes)
        return x_classes, y_classes

    def _train_test_split(self,
                          X,
                          y,
                          test_size:float=0.2, 
                          random_state:int=None) -> None:
        trains = {}
        tests = {}
        for i, cat in enumerate(self.cats):
            X = self.X[cat]
            y = self.y[cat]
            if test_size:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=test_size,
                                                                    random_state=random_state)
            else:
                X_train = X
                X_test = None
                y_train = y
                y_test = None

            trains.update({cat: (X_train, y_train)})
            tests.update({cat: (X_test, y_test)})
        self.trains = trains
        self.tests = tests
            
    def transform(self):
        return self._make_data()
            
    def make_train_test_data(self):
        X_transf, y_transf = self.transform()
        if self.split_train_test:
            self.train_test_split(X_transf, y_transf, 
                                  test_size=self.test_size, 
                                  random_state=self.random_state)
        else:
            self.train_test_split(X_transf, y_transf, 
                                  test_size=False,
                                  random_state=self.random_state)

    def feed_cat_Xy(self, cat:str, train:bool=True):
        if train:
            X, y = self.trains[cat]
        else:
            X, y = self.tests[cat]
        return X, y
        
