from setuptools import find_packages, setup

setup(name='brmp',
      packages=find_packages(),
      install_requires=[
        'pandas',
        'pyro-ppl>=0.4.1',
        'numpyro>=0.2.0',
      ],
      extras_require={
          'test': ['flake8', 'pytest'],
          'docs': ['sphinx', 'sphinx-rtd-theme'],
      })
