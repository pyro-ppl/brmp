from setuptools import find_packages, setup

setup(name='brmp',
      packages=find_packages(),
      license='Apache License 2.0',
      install_requires=[
        'pandas',
        'pyro-ppl>=1.0.0',
        'numpyro>=0.2.1',
      ],
      extras_require={
          'test': ['flake8', 'pytest'],
          'docs': ['sphinx', 'sphinx-rtd-theme'],
      })
