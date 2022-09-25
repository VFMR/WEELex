from setuptools import setup, find_packages

setup(
        name='weelex',
        version='0.0.1',
        description='A module for WEELex',
        author='Valentin Reich',
        packages=['weelex', 'batchprocessing', 'cluster_tfidf'],
        install_requires=[
              'numpy==1.21.2',
              'seaborn==0.11.2',
              'matplotlib==3.5.0',
              'scipy==1.7.3',
              'scikit-learn==1.0.2',
              'gensim==4.0.1',
              'pandas==1.3.5',
              'spacy==2.3.5',
              'tqdm==4.62.3',
              # 'python==3.9.7',
              'nltk==3.6.5',
            ],
        package_dir={
            '': '.',
            'batchprocessing': './batchprocessing',
            'cluster_tfidf': './cluster_tfidf/cluster_tfidf'
            }
        )
