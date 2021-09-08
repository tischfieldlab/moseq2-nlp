from setuptools import setup, find_packages


setup(
    name='moseq2-nlp',
    description='Interrogating Moseq data using a NLP-based approach',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.19.2',
        'tqdm==4.36.1',
        'gensim==4.0.1',
        'python-Levenshtein',
        'ConfigArgParse==1.5.2',
        'scikit_learn==0.24.2'
        'h5py',
        'moseq2-viz @ git+https://github.com/tischfieldlab/moseq2-viz.git@master'
    ],
    #entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
