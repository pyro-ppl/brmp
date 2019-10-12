from setuptools import setup, find_packages

setup(name='brmp',
      packages=find_packages(),
      install_requires=[
        'pandas',
        'pyro-ppl>=0.4.1',
        'numpyro>=0.2.0',
      ],
      extras_require={
        'test': ['flake8', 'pytest']
      })
