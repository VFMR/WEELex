from setuptools import setup, find_packages

setup(
        name='weelex',
        version='0.0.1',
        description='A module for WEELex',
        author='Valentin Reich',
        packages=['weelex'],
        install_requires=[
              'numpy',
              'seaborn==0.11.2',
              'matplotlib==3.5.0',
              'scipy==1.7.3',
              'scikit-learn==1.0.2',
              'gensim',
              'pandas==1.3.5',
              'spacy',
              'tqdm==4.62.3',
              'nltk>=3.6.6',
              'batchprocessing @ git+https://github.com/VFMR/batchprocessing.git#egg=batchprocessing-0.1',
              'cluster_tfidf @ git+https://github.com/VFMR/cluster_tfidf.git#egg=cluster_tfidf.1'
            ],
        package_dir={
            '': '.',
            }
        )
