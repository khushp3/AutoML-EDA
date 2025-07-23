from setuptools import setup, find_packages

setup(
    name='automl',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'jupyter',
        'featuretools',
    ],
    author='Khush',
    description='Automated EDA and ML-ready preprocessing tool',
)
