# Testing instructions

## Installation:

### Development install:
In case you actively modify the weelex package.
Changes to the package will be automatically reflected in the package installation.
```
conda create --name weelex python pip
conda activate weelex
python -m pip install -e .[dev]
```


## Running Tests:
`pytest --doctest-modules weelex/ tests/myunittests.py`

## Updating documentation
`./docs/make.bat html`
