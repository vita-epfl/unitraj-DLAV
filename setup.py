from setuptools import setup, find_packages

setup(
    name='motionnet',
    version='0.1.0',
    author='Lan Feng',
    author_email='lan.feng@epfl.ch',
    description='A unified framework for trajectory prediction.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vita-epfl/unitraj-DLAV',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
