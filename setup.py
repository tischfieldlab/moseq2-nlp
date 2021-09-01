from setuptools import setup, find_packages
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')


setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.1.3',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['tqdm', 'matplotlib', 'click', 'dtaidistance', 'sklearn',
                      'ruamel.yaml>=0.15.0', 'seaborn', 'psutil',
                      'pandas', 'networkx', 'numpy'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
