conda create -n predspot python=3.7
conda activate predspot
pip install pandas statsmodels==0.10.2 geojsoncontour stldecompose scikit-learn matplotlib descartes
conda install -y rtree geopandas
pip install .
