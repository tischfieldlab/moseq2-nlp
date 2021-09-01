from setuptools import setup, find_packages


setup(
    name='moseq2-nlp',
    description='Interrogating Moseq data using a NLP-based approach',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    install_requires=[],
    python_requires='>=3.6',
    #entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
