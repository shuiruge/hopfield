from setuptools import setup, find_packages


NAME = 'hopfield'
DESCRIPTION = 'Hopfield networks'
VERSION = '0.0.1'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=[
        'dat.*', 'dat']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3+',
    ],
    zip_safe=False,
)
