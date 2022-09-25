import time
import pickle
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import spacy
from nltk.corpus import stopwords

SPACY_MODEL = 'de_core_news_lg'


def remove_urls(texts):
    """
    Function to remove URLs found in strings via regular expressions

    Args:
        texts (pandas.Series): Array with strings

    Returns:
        (pandas.Series): Array with urls removed
    """
    url_regex = '(https?\:\/\/)?www[1-9]?\.[^ ]'
    html_regex = '(<?a )?href="[^"]*"( rel="[^"]*")?( target="[^"]*")?>'

    # start by replacing some false positives:
    texts_clean = texts.str.replace('^\.{3}', '', regex=True)\
                       .str.replace('\.{3}', '.', regex=True)

    texts_clean = texts_clean.str.replace(html_regex, '', regex=True)

    # the "proper" url regex can easily match typos (i.e. missing spaces)
    # hence, a less robust url search is implemented
    texts_clean = texts_clean.str.replace(url_regex, '<URL>', regex=True)
    texts_clean = texts_clean.str.replace('</a>', '', regex=False)
    return texts_clean


class TfidfCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X, y=None):
        return tfidf_cleanup(X)


def tfidf_cleanup(texts):
    parties = [
        'CDU', 'Cdu', 'cdu',
        'CSU', 'Csu', 'csu',
        'AFD', 'Afd', 'AfD', 'afd',
        'SPD', 'Spd', 'spd',
        'Grünen', 'Bündnis90',
        'Piratenpartei',
        'FDP', 'Fdp', 'fdp',
        'NPD', 'Npd', 'npd',
        'ÖDP', 'Ödp', 'ödp',
        'FPÖ', 'Fpö', 'fpö',
        'SPÖ', 'Spö', 'spö',
        'ÖVP', 'Övp', 'övp',
        'die Linke',
        'Republikaner', 'Republicans', 'republikaner', 'republicans',
        'Demokraten', 'Democrats', 'demokraten', 'democrats',
    ]

    teams = [
        'Borussia', 'borussia', 'borussen', 'Borussen', 'BVB', 'bvb'
        'FCB', 'FC Bayern', 'fc bayern', 'fcb',
        'Werder Bremen', 'Werder', 'werder',
        ' TSV', ' tsv', 'Eintracht', 'VfB', 'Hertha BSC', 'hertha bsc',
        'hertha', 'Arminia Bielefeld', 'EffZee Köln',
    ]

    replacement_dict = {
        'FIFA': 'Fußball Verband',
        'fifa': 'Fußball Verband',
        'DFB': 'Fußball Verband',
        'dfb': 'Fußball Verband',
        'BuLi': 'Bundesliga',
        'IOC': 'Olympisches Komitee',
        'BuLa': 'Bundesland',
        'NRW': 'Bundesland',
        'kmh': 'Geschwindigkeit',
        'km/h': 'Geschwindigkeit',
        '5km': '5 Kilometer',
        '5 km': '5 Kilometer',
        '5 Km': '5 Kilometer',
        'kwh': 'Kilowatt',
        'kWh': 'Kilowatt',
        'kcal': 'Kalorien',
        ' OS ': ' Betriebssystem ',
        '5kg': '5 Kilogramm',
        '5 kg': '5 Kilogramm',
        'LKA': 'Kriminalamt',
        'BKA': 'Kriminalamt',
        'PLZ': 'Postleitzahl',
        'Tesla': 'Auto',
        '5EUR': '5 Euro',
        '5 EUR': '5 Euro',
        '5 Eur': '5 Euro',
        ' EUR ': ' Euro ',
        ' EUR.': ' Euro.',
        ' EUR,': ' Euro,',
        '5mm': '5 Milimeter',
        '5 mm': '5 Milimeter',
        '5 ltr': '5 Liter',
        '5 Ltr': '5 Liter',
        'D Mark': 'Euro',
        'D-Mark': 'Euro',
        'DMark': 'Euro',
        'SUV': 'Auto',
        ' IT ': ' Informationstechnologie ',
        'NGO': 'Organisation',
        'MWST': 'Mehrwertsteuer',
        'mwst': 'Mehrwertsteuer',
        'MwSt': 'Mehrwertsteuer',
        'den Nagel auf den Kopf': 'richtig',
        'ZDF': 'Fernsehsender',
        'ARD': 'Fernsehsender',
        'RTL': 'Fernsehsender',
        'Pkw': 'Auto',
        'Lkw': 'Auto',
        'PKW': 'Auto',
        'LKW': 'Auto',
        'Know-How': 'Wissen',
        'Knowhow': 'Wissen',
        'ã¤': 'ä',
        'ã¼': 'ü',
        'ã¶': 'ö',
        #'Ã\\x9c': 'Ü',´
        'ã\x9f': 'ß',
        'ã\\x9f': 'ß',
        'Ã\x9c': 'Ü',
        'Ã\\x9c': 'Ü',
        'Ã¼': 'Ü',
        'Ã¶': 'Ö' ,
        'Ã\x96': 'Ö',
        'Ã\\x96': 'Ö',
        r'Ã\x84': 'Ä',
        'Ã\x84': 'Ä',
        'Ã\\x84': 'Ä',
    }

    removes = [
        '&gt,',
    ]

    texts = texts.apply(lambda x: np.str_(x))  # otherwise encoding issues.

    # replace some of the most prominent politicians names with their profession
    # this is done 1) with first and last name and 2) only with last name to
    # avoid having gramatically incorrect remaining texts like
    # "Angela Kanzlerin".
    # We need mostly functioning grammar for the tfidf-analyzer to do
    # successful part of speech tagging and named entity recognition.
    texts = texts.str.replace('Angela Merkel', 'Kanzlerin')
    texts = texts.str.replace('Donald Trump', 'Präsident')
    texts = texts.str.replace('Merkel', 'Kanzlerin')
    texts = texts.str.replace('Trump', 'Präsident')

    texts = texts.str.replace('[Aa]+ber', 'aber', regex=True)
    texts = texts.str.replace('<NUM>', '5', regex=False)
    texts = texts.str.replace('5 [Mm]rd\.?', '5 Milliarden', regex=True)
    texts = texts.str.replace('5 [Mm]io\.?', '5 Millionen', regex=True)
    texts = texts.str.replace(r'5\\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(r'5 \\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(r'5T\\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(r'5T \\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(r'5M\\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(r'5M \\xc5\\x5', '5 Euro', regex=False)
    texts = texts.str.replace(
        r'Millionen \\xc5\\x5', 'Millionen Euro', regex=False)
    texts = texts.str.replace(
        r'Milliarden \\xc5\\x5', 'Millionen Euro', regex=False)
    texts = texts.str.replace('[MSD]eine Frau ', 'Ehefrau', regex=True)
    texts = texts.str.replace(' [msd]eine Frau', ' Ehefrau', regex=True)
    texts = texts.str.replace('[MSD]ein Mann ', 'Ehemann', regex=True)
    texts = texts.str.replace(' [msd]ein Mann', ' Ehemann', regex=True)
    texts = texts.str.replace('Ex-Mann', 'Ehemann', regex=False)  #
    texts = texts.str.replace('Ex-Männer', 'Ehemänner', regex=False)
    texts = texts.str.replace('Ex-Frau', 'Ehefrau', regex=False)

    texts = texts.str.replace('Ihre Frauen ', 'Ehefrau', regex=True)
    texts = texts.str.replace(' ihre Frauen', ' Ehefrau', regex=True)
    texts = texts.str.replace('Ihre Männer ', 'Ehemann', regex=True)
    texts = texts.str.replace(' ihre Männer', ' Ehemann', regex=True)

    texts = texts.str.replace('Ihre Frau ', 'Ehefrau', regex=True)
    texts = texts.str.replace(' ihre Frau', ' Ehefrau', regex=True)
    texts = texts.str.replace('Ihr Mann ', 'Ehemann', regex=True)
    texts = texts.str.replace(' ihr Mann', ' Ehemann', regex=True)

    texts = texts.str.replace('xcx', '', regex=False)\
                 .str.replace('Xcx', '', regex=False)\
                 .str.replace('XCX', '', regex=False)

    texts = texts.str.replace('GroKo', 'Koalition', regex=False)
    texts = texts.str.replace('groko', 'Koalition', regex=False)
    texts = texts.str.replace('Groko', 'Koalition', regex=False)

    # capture remaining URLs
    texts = remove_urls(texts)

    # split words on hyphens
    texts = texts.str.replace('-', ' ', regex=False)

    for party in parties:
        texts = texts.str.replace(party, 'Partei', regex=False)

    for team in teams:
        texts = texts.str.replace(team, 'Verein', regex=False)

    for key, value in replacement_dict.items():
        texts = texts.str.replace(key, value, regex=False)

    for term in removes:
        texts = texts.str.replace(term, '', regex=False)

    return texts


class BasicTfidf:
    def __init__(self,
                 stopwords_file=None,
                 relevant_pos=['ADJ', 'ADV', 'NOUN', 'VERB'],
                 min_df=5,
                 max_df=0.95,
                 spacy_model=SPACY_MODEL):
        self.stopwords_file = stopwords_file
        self.min_df = min_df
        self.max_df = max_df
        self.spacy_model_name = spacy_model
        self.spacy_model = spacy.load(spacy_model)
        self.sw = self._retrieve_stopwords(
            path=stopwords_file
        )
        self.relevant_pos = relevant_pos
        self._is_fit = False

    @staticmethod
    def _retrieve_stopwords(path):
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                custom_sw = list(
                    set([x.replace('\n', '').strip() for x in f.readlines()]))
        else:
            custom_sw = []

        sw = stopwords.words('german')
        sw = sw + [' ', ''] + custom_sw
        return sw

    def _get_params_from_child(self):
        pass

    def _analyzer(self, comment):

        # setting up things for speedup
        relevant_pos = self.relevant_pos
        capitalize = str.capitalize
        lower = str.lower
        pattern = '[\-\.\,\#\'\"\+\~\*\=\?\!\$\§\%\&\/\(\)\[\]\{\}\<\>\|\^\°\_\:\\\;]'
        tokens = []
        append = tokens.append
        sw = self.sw

        # tokenization
        doc = self.spacy_model(comment)

        # selecting words:
        for x in doc:
            pos = x.pos_
            ent_type = x.ent_type_
            if pos in relevant_pos and ent_type not in ['PER', 'LOC']:
                if not (ent_type=='ADJ' and pos=='MISC'):  #  capture nationalities
                    word = re.sub(pattern, '', x.lemma_)
                    if pos=='NOUN':
                        word = capitalize(word)
                    else:
                        word = lower(word)
                    if word not in sw and not 'xcx' in word and not 'Xcx' in word and not '5' in word:
                        append(word)

        return tokens

    def load(self, path):
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def fit(self, X, y=None):

        print('Fit vectorizer')
        start = time.time()

        vectorizer_only = TfidfVectorizer(
                analyzer=self._analyzer,
                #stop_words=stopwords.words('german'),  # not used if analyzer is callable
                min_df=self.min_df,
                max_df=self.max_df,
                lowercase=False,)

        self.vectorizer = Pipeline(steps=[
            ('cleaner', TfidfCleaner()),
            ('vectorizer', vectorizer_only)
        ])

        self.vectorizer.fit(X)
        self._is_fit = True

        end = time.time()
        print('Time to vectorize: {:0.2f} minutes'.format((end-start)/60))

        self._get_params_from_child()

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def check_fit(self):
        return self._is_fit

    def save(self, path):
        # exporting the different results from vectorization:
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)  # vectorizer

    # for compatibility, include all the other methods of the TfidfVectorizer class
    def get_feature_names(self):
        return self.vectorizer.steps[-1].get_feature_names()

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.steps[-1].get_feature_names(input_features=input_features)

    def get_params(self, deep=True):
        dct = self.vectorizer.steps[-1].get_params(deep=deep)
        dct.update({'spacy_model': self.spacy_model_name,
                    'stopwords_file': self.stopwords_file,
                    'relevant_pos': self.relevant_pos})
        return dct

    def get_stop_words(self):
        return self.sw

    def inverse_transform(self, X):
        return self.vectorizer.steps[-1].inverse_transform(X)

    def set_params(self, **params):
        if 'spacy_model' in params.keys():
            self.spacy_model_name = params['spacy_model']
            self.spacy_model = spacy.load(params['spacy_model'])
            del params['spacy_model']

        if 'stopwords_file' in params.keys():
            self.stopwords_file = params['stopwords_file']
            self.sw = self._retrieve_stopwords(path=self.stopwords_file)
            del params['stopwords_file']

        if 'relevant_pos' in params.keys():
            self.relevant_pos = params['relevant_pos']
            del params['relevant_pos']

        self.vectorizer.steps[-1].set_params(**params)

    def build_analyzer(self):
        """Calls TfidfVectorizer build_analyzer() method.
        Does not include the proprocessing thus far.
        """
        return self.vectorizer.steps[-1].build_analyzer()

    def build_preprocessor(self):
        return self.vectorizer.steps[0].transform

    def build_tokenizer(self):
        """Calls TfidfVectorizer build_tokenizer() method.
        Does not include the preprocessing thus far.
        """
        return self.vectorizer.steps[-1].build_tokenizer()

    def decode(self, doc):
        return self.vectorizer.steps[-1].decode(doc)

    @property
    def vocabulary_(self):
        return self.vectorizer.steps[-1][1].vocabulary_

    @property
    def fixed_vocabulary_(self):
        return self.vectorizer.steps[-1][1].fixed_vocabulary_

    @property
    def idf_(self):
        return self.vectorizer.steps[-1][1].idf_

    @property
    def stop_words_(self):
        return list(set(
            self.sw + list(self.vectorizer.steps[-1][1].stop_words_)
            ))
