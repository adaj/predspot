conda create -n predspot python=3.8
conda activate predspot
conda install -y rtree geopandas  # if doesnt work, do: `conda clean --all`
pip install pandas scikit-learn statsmodels matplotlib seaborn geojsoncontour descartes contextily
# if statsmodels import problem: pip install --upgrade --force-reinstall statsmodels
pip install .
