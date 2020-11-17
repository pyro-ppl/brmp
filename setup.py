from setuptools import find_packages, setup

setup(name='brmp',
      packages=find_packages(),
      license='Apache License 2.0',
      install_requires=[
        'pandas',
        'pyro-ppl>=1.4.0',
        'numpyro>=0.4.0',
      ],
      extras_require={
          'test': ['flake8', 'pytest'],
          'docs': ['sphinx', 'sphinx-rtd-theme'],
      },
      python_requires='>=3.6',
      )
