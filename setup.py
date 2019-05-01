from setuptools import setup

setup(
    name='torchtest',
    version='0.5',
    packages=['torchtest'],
    license='GNU Affero General Public License v3 or later (AGPLv3+)',
    long_description=open('README.adoc').read(),
    install_requires=['torch'],
)
