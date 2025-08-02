from setuptools import setup, find_packages

# also change in version.py
VERSION = '0.2.0'
DESCRIPTION = "Bulk backtesting for Jesse - now with Optuna support"

REQUIRED_PACKAGES = [
    'jesse',
    'pyyaml',
    'joblib',
    'psutil',
    'optuna'
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='jesse-bulk',
    version=VERSION,
    author="cryptocoinserver",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cryptocoinserver/jesse-bulk",
    install_requires=REQUIRED_PACKAGES,
    entry_points='''
        [console_scripts]
        jesse-bulk=jesse_bulk.__init__:cli
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
