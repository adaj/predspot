conda create -n predspot-ufrn python=3.7
conda activate predspot-ufrn
pip install pandas statsmodels==0.10.2 geojsoncontour stldecompose scikit-learn matplotlib descartes
conda install -y geopandas
pip install .
