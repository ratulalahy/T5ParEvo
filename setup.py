from setuptools import find_packages, setup

setup(
    name='T5ParEvo',
    packages=find_packages(where='T5ParEvo'),
    version='0.1.1',
    description='BioMedical sentence paraphrasing by finetuning T5 model',
    author='ratulalahy',
    license='MIT',
)
