from setuptools import setup, find_packages

setup(
    name='syntheval',
    version='1.7.0',
    packages=['syntheval'],
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'SynthEval = syntheval.__main__:main',
        ],
    },
)