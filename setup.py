from setuptools import setup, find_packages

setup(
    name='fluxtune',
    version='0.1.1',
    description='Flux tune is a library intended to implemnet various online machine learning algorithms and agents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Pradeep8802/fluxtune',  
    author='SAI PRADEEP',
    author_email='saipradeepirgp@gmail.com',
    license='MIT', 
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0', 
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7', 
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7', 
)
