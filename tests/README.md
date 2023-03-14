# Testing instructions

## Installation:

### Development install:
In case you actively modify the weelex package.
Changes to the package will be automatically reflected in the package installation.
```
conda create --name weelex python pip
conda activate weelex
python -m spacy download de_core_news_lg
python -m pip install -e .
```


## Running Tests:
`pytest --doctest-modules --ignore=batchprocessing/tests/ tests/myunittests.py weelex/`

