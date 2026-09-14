from setuptools import setup

setup(
    name='predspot',
    version='0.2.0',
    description="Predicting crime hotspots with machine learning",
    url='https://github.com/adaj/predspot',
    author="Adelson Araujo",
    author_email='adelson.dias@gmail.com',
    packages=['predspot'],
    install_requires=[
        'numpy>=1.24',
        'pandas>=2.2',
        'geopandas>=1.0',
        'shapely>=2.0',
        'scipy>=1.10',
        'scikit-learn>=1.3',
        'statsmodels>=0.14',
        'matplotlib>=3.7',
    ],
    extras_require={'contour': ['geojsoncontour']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: BSD 3-Clause License'
    ],
    python_requires='>=3.10',
)
