from setuptools import setup, find_packages

setup(
    name='energy_pricing',
    version='1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'xgboost'
    ],
    entry_points={
        'console_scripts': [
            'energy_pricing=main:main', 
        ],
    },
)
