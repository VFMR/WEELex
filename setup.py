from setuptools import setup, find_packages

setup(
    name="weelex",
    version="0.1.0",
    description="A module for WEELex",
    author="Valentin Reich",
    packages=["weelex"],
    install_requires=[
        "numpy",
        "seaborn==0.11.2",
        "matplotlib==3.5.0",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
        "gensim",
        "pandas==1.3.5",
        "spacy",
        "tqdm==4.62.3",
        "nltk>=3.6.6",
        "batchprocessing @ git+https://github.com/VFMR/batchprocessing.git@afb5985ea4297adb9323280b7b3bfeb035124e94#egg=batchprocessing-0.1",
        "cluster_tfidf @ git+https://github.com/VFMR/cluster_tfidf.git#egg=cluster_tfidf.1",
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "sphinx_rtd_theme", "black", "pylint", "pre-commit"]
    },
    package_dir={
        "": ".",
    },
)
