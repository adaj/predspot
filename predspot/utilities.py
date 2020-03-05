import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geojsoncontour
from numpy import zeros, linspace

def contour_geojson(y, lonv, latv, cmin, cmax):
    Z = zeros(lonv.shape[0]*lonv.shape[1]) - 999
    Z[y.index] = y.values
    Z = Z.reshape(lonv.shape)
    fig, axes = plt.subplots()
    contourf = axes.contourf(lonv, latv, Z,
                            levels=linspace(cmin, cmax, 25),
                            cmap='Spectral_r')
    geojson = geojsoncontour.contourf_to_geojson(contourf=contourf, fill_opacity=0.5)
    plt.close(fig)
    return geojson
