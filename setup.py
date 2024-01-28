from setuptools import setup
from setuptools import find_packages
from glob import glob
from os.path import splitext
from os.path import basename
import versioneer

setup(
    name = 'nlp_diplomacy',
    description = 'Natural Lenguage Procesing ',
    url = ' ',
    author = 'Valentina Vasquez Echavarria',
    author_email = 'valevasq@bancolombia.com.co',
    license = '...',
    packages = find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires = [
        'orquestador2',
        'scikit-learn',
        'transformers',
        'tensorflow',
        'openpyxl',
        'seaborn',
        'matplotlib',
        'nltk'
    ],
    include_package_data = True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
