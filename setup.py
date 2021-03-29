from setuptools import setup

setup(
    name='predspot',
    version='0.1.3',
    description="Predicting crime hotspots with machine learning",
    url='https://github.com/adaj/predspot',
    author="Adelson Araujo",
    author_email='adelson.dias@gmail.com',
    # packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'statsmodels==0.12.1',
        'rtree',
        'contextily',
        'geojsoncontour',
        'geopandas'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.7'
)
