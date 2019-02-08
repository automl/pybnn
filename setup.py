from setuptools import setup, find_packages

setup(
    name='pybnn',
    version='0.0.5',
    description='Simple python framework for Bayesian neural networks',
    author='Aaron Klein, Moritz Freidank',
    author_email='kleinaa@cs.uni-freiburg.de',
    url="https://github.com/automl/pybnn",
    license='BSD 3-Clause License',
    classifiers=['Development Status :: 4 - Beta'],
    packages=find_packages(),
    python_requires='>=3',
    install_requires=['torch', 'torchvision', 'numpy', 'emcee', 'scipy'],
    extras_require={},
    keywords=['python', 'Bayesian', 'neural networks'],
)
