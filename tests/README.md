# Testing instructions

## Installation:

### Development install:
In case you actively modify the weelex package.
Changes to the package will be automatically reflected in the package installation.
```
conda create --name weelex python pip
conda activate weelex
python -m pip install -e .
python -m spacy download de_core_news_lg
conda install pytest
conda install sphinx
conda install sphinx_rtd_theme
```


## Running Tests:
`pytest --doctest-modules --ignore=batchprocessing/tests/ tests/myunittests.py weelex/`

## Updating documentation
`./docs/make.bat html`


