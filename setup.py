from setuptools import setup, find_packages
import .pySNOM

setup(
    name='pySNOM',
    version=pySNOM.__version__,
    description='Scanning Near-Field Optical Microscopy (SNOM) analysis tools',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/ngergihun/pySNOM',
    author='Gergely Nemeth, Ferenc Borondics',
    author_email='ngergihun@gmail.com, borondics@gmail.com',
    license='CC BY-NC-SA 4.0 DEED',
    packages=find_packages(),
    install_requires=['scipy',
                      'numpy',
                      'gwyfile',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
    ],
)
