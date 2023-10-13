from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'matplotlib',
    'ray==2.3.1', 
    'numpy',
    'pandas',
    'requests',  # used in c4.5
    'scipy',
    'pettingzoo>=1.24',  # 0.23+ only works on py3.6+
    'tqdm',  # used in BART
]

extra_deps = [
    'cvxpy',  # optionally requires cvxpy for slim
    'corels',  # optionally requires corels for optimalrulelistclassifier
    'gosdt-deprecated',  # optionally requires gosdt for optimaltreeclassifier
    'irf',  # optionally require irf for iterativeRandomForestClassifier
]

setuptools.setup(
    name="AIEconomist2",
    version="1.4.1",
    author="Subarno Sen",
    author_email="subarnos@live.com",
    description="Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lordfiftyfive/Ai-Economist2",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    extras_require={
        'dev': [
            'dvu',
            'gdown',
            # 'irf',
            'jupyter',
            'jupytext',
            'matplotlib',
            # 'pdoc3',  # for building docs
            'pytest',
            'pytest-cov',
            # 'seaborn',  # in bartpy.diagnostics.features
            'slurmpy',
            # 'statsmodels', # in bartpy.diagnostics.diagnostics
            # 'torch',  # for neural-net-integrated models
            'tqdm',
            'pmlb',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
