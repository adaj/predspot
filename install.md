```
$ conda create -n predspot python=3.8

$ conda activate predspot

$ conda install -y rtree geopandas  (if this doesnt work, do: `conda clean --all`)

$ pip install pandas scikit-learn statsmodels==0.12.1 matplotlib seaborn geojsoncontour descartes contextily

$ git clone https://github.com/adaj/predspot.git

$ cd predspot

$ pip install . 
```
