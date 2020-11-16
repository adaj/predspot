conda create -n predspot python=3.8
conda activate predspot
conda install -y rtree geopandas  # if doesnt work, do: `conda clean --all`
pip install pandas statsmodels==0.10.2 geojsoncontour stldecompose scikit-learn matplotlib descartes
pip install .
