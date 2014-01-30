try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Jet Analysis framework in Python using PyROOT bindings',
    'author': 'Giordon Stark',
    'url': 'http://www.giordonstark.com/',
    'download_url': 'Where to download it.',
    'author_email': 'kratsg@cern.ch',
    'version': '0.1',
    'install_requires': ['nose','root_numpy'],
    'packages': ['atlas_jets'],
    'scripts': [],
    'name': 'atlas_jets',
    'classifiers': ['Programming Language :: Python',\
                    'Programming Language :: Python :: 2.7',\
                    'Topic :: Scientific/Engineering :: Physics',\
                    'Operating System :: OS Independent',\
                    'Development Status :: 4 - Beta',\
                    'Intended Audience :: Science/Research']
}

setup(**config)
