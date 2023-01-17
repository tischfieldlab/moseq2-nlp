from setuptools import setup, find_packages


setup(
    name='moseq2-nlp',
    description='Interrogating Moseq data using a NLP-based approach',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'brown-clustering',
        'tqdm',
        'notebook',
        'gensim==4.0.1',
        'python-Levenshtein',
        'scikit_learn==0.24.2',
        'h5py',
        'click',
        'click-option-group',
        'ruamel.yaml',
        'moseq2-viz', # @ git+https://github.com/tischfieldlab/moseq2-viz.git@master'
        'wordcloud',
        'eli5'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'pydocstyle',
            'mypy',
            'black'
        ]
    },
    entry_points={'console_scripts': ['moseq2-nlp = moseq2_nlp.cli:cli']}
)
