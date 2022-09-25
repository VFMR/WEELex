# Word Embedding Enhanced Lexica

## Installation:
### Regular install:
```
conda create --name weelex python pip
conda activate weelex
python -m pip install .
```

### Development install:
In case you actively modify the weelex package.
Changes to the package will be automatically reflected in the package installation.
```
conda create --name weelex python pip
conda activate weelex
python -m pip install -e .
```

### Troubleshooting:
#### import errors (Microsoft Visual Studio Build Tools required).
Install the packages that require a c compiler with conda first:
```
conda create --name weelex python pip gensim=4.0.1 spacy=2.3.5
conda activate weelex
python -m pip install .
```

## Running Tests:
`pytest --doctest-modules --ignore=batchprocessing/tests/ tests/myunittests.py weelex/`

