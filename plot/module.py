from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy
import copy
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import detrend
import scipy.stats
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings(action="ignore")
###################################################################################
## Plotting functions
# Plotting shaded countour figure
def shaded(da_cs, lon_range, lat_range, 
           dlon=30, dlat=30, scale=0.7, title="",
           cb_levels=21, cmap=plt.cm.RdBu_r, cb_label="", cb_pad=0.02, cb_width=20, cb_ticks=None):
    '''
    #Descriptions for parameters#
    da_cs: dataarray that you want to draw. It must have 2D shape.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    label: the label of the colorbar.
    cb_levels: set the levels of contours.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cmap: set the color map of shaded contour.
    cb_pad: the distance between the figure and the colorbar. it's proportional to the horizontal size of figure
    cb_width: the width of colorbar. when it is bigger, the width of color bar is smaller. (1/cb_width)
    cb_ticks: set the ticks of colorbar.
    '''
    # Rename instances
    da = da_cs
    levels = cb_levels
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                      dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)

    # Draw a shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cs = ax.contourf(LON, LAT, da, transform=data_crs,
                     cmap=cmap, levels=levels, extend="both")
    
    cb = plt.colorbar(cs, orientation="vertical", pad=cb_pad, aspect=cb_width, ticks=cb_ticks)
    cb.set_label(cb_label)
    # Draw a contour figure over the shaded figure
    ax.contour(LON, LAT, da, transform=data_crs,
               levels=levels, colors="k", linewidths=0.5)

# Plotting countour line figure
def contour(da_cl, lon_range, lat_range, 
            dlon=30, dlat=30, scale=0.7, title="",
            cl_label=True, cl_levels=21, cl_linewidth=0.5, cl_labelfmt="%1.0f"):
    '''
    #Descriptions for parameters#
    da_cl: dataarray that you want to draw. It must have 2D shape.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cl_label: the label of the colorbar.
    cl_levels: set the levels of contours.
    cl_label: if it is True, the label of line contour is added among the line contours. (default value: Ture)
    cl_levels: set the levels of line contours.
    cl_linewidth: set the line width of line contours.
    cl_labelfmt: set the format of line contour labels.
    '''
    # Rename instances
    da = da_cl
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                      dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)

    # Draw a contour line figure over the shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cl = ax.contour(LON, LAT, da, transform=data_crs,
                    levels=cl_levels, colors="k", linewidths=cl_linewidth)
    if cl_label == True:
        plt.clabel(cl, inline=True, fmt=cl_labelfmt, fontsize=10, colors="k")

# Plotting shaded and countour line figure
def contour_shaded(da_cs, da_cl, lon_range, lat_range,
                   dlon=30, dlat=30, scale=0.7, title="",
                   cb_label="", cb_levels=21, cmap=plt.cm.RdBu_r, cb_pad=0.02, cb_width=20, cb_ticks=None,
                   cl_label=True, cl_levels=21, cl_linewidth=0.5, cl_labelfmt="%1.0f"):
    '''
    #Descriptions for parameters#
    da_cs: dataarray that you want to draw shaded contour. It must have 2D shape.
    da_cl: dataarray that you want to draw line contour. It must have 2D shape.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cb_label: the label of the colorbar.
    cl_label: if it is True, the label of line contour is added among the line contours. (default value: Ture)
    cb_levels: set the levels of shaded contours.
    cl_levels: set the levels of line contours.
    cmap: set the color map of shaded contour.
    cl_linewidth: set the line width of line contours.
    cl_labelfmt: set the format of line contour labels.
    cb_pad: the distance between the figure and the colorbar. it's proportional to the horizontal size of figure
    cb_width: the width of colorbar. when it is bigger, the width of color bar is smaller. (1/cb_width)
    cb_ticks: set the ticks of colorbar.
    '''
    # Rename instances
    da = da_cs
    da2 = da_cl
    levels = cb_levels
    levels2 = cl_levels
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    da2 = da2.sortby(da2.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                      dims=["lat", "lon"])
    lon_idx = da2.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da2, coord=da2.lon, axis=lon_idx)
    da2 = xr.DataArray(wrap_data, coords=[da2.lat, wrap_lon],
                       dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        da2.coords["lon"] = (da2.coords["lon"] + 180) % 360 - 180
        da2 = da2.sortby(da2.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)
    da2 = da2.sel(lon=lon_range, lat=lat_range)

    # Draw a shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cs = ax.contourf(LON, LAT, da, transform=data_crs,
                     cmap=cmap, levels=levels, extend="both")
    cb = plt.colorbar(cs, orientation="vertical", pad=cb_pad, aspect=cb_width, ticks=cb_ticks)
    cb.set_label(cb_label)
    # Draw a contour line figure over the shaded figure
    LON, LAT = np.meshgrid(da2.lon, da2.lat)
    cl = ax.contour(LON, LAT, da2, transform=data_crs,
                    levels=levels2, colors="k", linewidths=cl_linewidth)
    if cl_label == True:
        plt.clabel(cl, inline=True, fmt=cl_labelfmt, fontsize=10, colors="k")


# Plotting quiver and shaded contour quiver
def quiver_shaded(da_cs, da_q, lon_range, lat_range, quiver_scale, quiver_width, key_length,
                  key_unit="", title="", cb_label="", dlon=30, dlat=30, scale=0.7,
                  cb_levels=21, cmap=plt.cm.RdBu_r, cb_pad=0.02, cb_width=20, cb_ticks=None,
                  step_x=3, step_y=2, key_loc_x=0.95, key_loc_y=1.05):
    '''
    #Descriptions for parameters#
    da_cs: dataarray that you want to draw as shaded contour figure. It must have 2D shape.
    da_q: list of dataarraies for quiver, [dataarray for x-component, dataarray for y-component]
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cb_label: the label of the colorbar.
    levels: set the levels of contours.
    cmap: set the color map of shaded contour.
    cb_pad: the distance between the figure and the colorbar. it's proportional to the horizontal size of figure
    cb_width: the width of colorbar. when it is bigger, the width of color bar is smaller. (1/cb_width)
    cb_ticks: set the ticks of colorbar.
    set_x, set_y: how many skip the data in the quiver datasets.
    quiver_scale: set the quiver scale. When it gets bigger, the length of quiver is shorter.
    quiver_width: set the quiver width.
    key_loc_x, key_loc_y: set the location of key quiver.
    key_length: set the standard value of key quiver.
    key_unit: set the unit of key quiver
    '''
    # Rename instances
    da0 = da_cs
    da1 = da_q[0]
    da2 = da_q[1]
    levels = cb_levels
    # Sorting latitude into (-90 ~ 90)
    da0 = da0.sortby(da0.lat)
    da1 = da1.sortby(da1.lat)
    da2 = da2.sortby(da2.lat)
    # Addcyclic
    lon_idx = da0.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da0, coord=da0.lon, axis=lon_idx)
    da0 = xr.DataArray(wrap_data, coords=[da0.lat, wrap_lon],
                       dims=["lat", "lon"])
    lon_idx = da1.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da1, coord=da1.lon, axis=lon_idx)
    da1 = xr.DataArray(wrap_data, coords=[da1.lat, wrap_lon],
                       dims=["lat", "lon"])
    lon_idx = da0.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da2, coord=da2.lon, axis=lon_idx)
    da2 = xr.DataArray(wrap_data, coords=[da2.lat, wrap_lon],
                       dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da0.coords["lon"] = (da0.coords["lon"] + 180) % 360 - 180
        da0 = da0.sortby(da0.lon)
        da1.coords["lon"] = (da1.coords["lon"] + 180) % 360 - 180
        da1 = da1.sortby(da1.lon)
        da2.coords["lon"] = (da2.coords["lon"] + 180) % 360 - 180
        da2 = da2.sortby(da2.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()

    # Draw a shaded figure
    da0 = da0.sel(lon=lon_range, lat=lat_range)
    LON, LAT = np.meshgrid(da0.lon, da0.lat)
    cs = ax.contourf(LON, LAT, da0, transform=data_crs,
                     cmap=cmap, levels=levels)  # , extend="both")
    cb = plt.colorbar(cs, orientation="vertical", pad=cb_pad, aspect=cb_width, ticks=cb_ticks)
    cb.set_label(cb_label)
    # Draw a vector figure
    da1 = da1.sel(lon=lon_range, lat=lat_range)
    da2 = da2.sel(lon=lon_range, lat=lat_range)
    LON, LAT = np.meshgrid(da1.lon, da1.lat)
    q = ax.quiver(LON[::step_x, ::step_y], LAT[::step_x, ::step_y],
                  da1[::step_x, ::step_y].values, da2[::step_x,
                                                      ::step_y].values,  # IndexError 때문에 .values 필수
                  transform=data_crs, scale=quiver_scale, width=quiver_width)
    plt.quiverkey(q, key_loc_x, key_loc_y, key_length,
                       label=f"{key_length} {key_unit}", labelpos="E", transform=proj)

# Plotting double contour line graph
def contour_contour(da_cl1, da_cl2, lon_range, lat_range,
                    dlon=30, dlat=30, scale=0.7, title="",
                    cl_label1=True, cl_levels1=21, cl_color1="k", cl_linewidth1=0.5, cl_labelfmt1="%1.0f",
                    cl_label2=True, cl_levels2=21, cl_color2="r", cl_linewidth2=0.5, cl_labelfmt2="%1.0f"):
    '''
    Double contour lines graph (overlapped)
    #Descriptions for parameters#
    da_cl1(or2): dataarray that you want to draw. It must have 2D shape.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cl_levels1(or2): set the levels of line contours.
    cl_label1(or2): if it is True, the label of line contour is added among the line contours. (default value: Ture)
    cl_linewidth1(or2): set the line width of line contours.
    cl_labelfmt1(or2): set the format of line contour labels.
    '''
    # data input                
    da = da_cl1
    da2 = da_cl2
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    da2 = da2.sortby(da2.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    lon_idx2 = da2.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    wrap_data2, wrap_lon2 = add_cyclic_point(da2, coord=da2.lon, axis=lon_idx2)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                    dims=["lat", "lon"])
    da2 = xr.DataArray(wrap_data2, coords=[da2.lat, wrap_lon2],
                    dims=["lat", "lon"])
    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da2.coords["lon"] = (da2.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        da2 = da2.sortby(da2.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                ylocs=np.arange(-90, 91, dlat),
                color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)
    da2 = da2.sel(lon=lon_range, lat=lat_range)

    # Draw a contour line figure over the shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cl = ax.contour(LON, LAT, da, transform=data_crs,
                    levels=cl_levels1, colors=cl_color1, linewidths=cl_linewidth1)
    if cl_label1 == True:
        plt.clabel(cl, inline=True, fmt=cl_labelfmt1, fontsize=10, colors=cl_color1)

    LON, LAT = np.meshgrid(da2.lon, da2.lat)
    cl2 = ax.contour(LON, LAT, da2, transform=data_crs,
                    levels=cl_levels2, colors=cl_color2, linewidth=cl_linewidth2)
    if cl_label2 == True:
        plt.clabel(cl2, inline=True, fmt=cl_labelfmt2, fontsize=10, colors=cl_color2)

# Plotting shaded countour and dotted figure
def hatch_shaded(da_cs, da_dot, lon_range, lat_range, 
                 dlon=30, dlat=30, scale=0.7, title="",
                 cb_levels=21, cmap=plt.cm.RdBu_r, cb_label="", cb_pad=0.02, cb_width=20, cb_ticks=None,
                 hatch_alpha=0, hatch_type="."):
    '''
    #Descriptions for parameters#
    da_cs: dataarray that you want to draw. It must have 2D shape.
    da_dot: dataarray for dotted figure. It must be True/False dtype.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    label: the label of the colorbar.
    cb_levels: set the levels of contours.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cmap: set the color map of shaded contour.
    cb_pad: the distance between the figure and the colorbar. it's proportional to the horizontal size of figure
    cb_width: the width of colorbar. when it is bigger, the width of color bar is smaller. (1/cb_width)
    cb_ticks: set the ticks of colorbar.
    hatch_alpha: set the transparency of hatches.
    hatch_type: default: '.', options: ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    '''
    # Rename instances
    levels = cb_levels
    da = da_cs
    da2 = da_dot
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    da2 = da2.sortby(da2.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    lon_idx2 = da2.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                      dims=["lat", "lon"])
    # longitude의 0도와 360도를 이어줌.
    wrap_data2, wrap_lon2 = add_cyclic_point(da2, coord=da2.lon, axis=lon_idx2)
    da2 = xr.DataArray(wrap_data2, coords=[da2.lat, wrap_lon2],
                       dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        da2.coords["lon"] = (da2.coords["lon"] + 180) % 360 - 180
        da2 = da2.sortby(da2.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)

    # Draw a shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cs = ax.contourf(LON, LAT, da, transform=data_crs,
                     cmap=cmap, levels=levels, extend="both")
    
    cb = plt.colorbar(cs, orientation="vertical", pad=cb_pad, aspect=cb_width, ticks=cb_ticks)
    cb.set_label(cb_label)
    # Draw a contour figure over the shaded figure
    ax.contour(LON, LAT, da, transform=data_crs,
               levels=levels, colors="k", linewidths=0.5)

    # Draw a hatches figure for True/False dataset
    levels_hatch = [-0.5, 0.5, 1.5]  # True or False 분류를 위해
    da2 = da2.sel(lon=lon_range, lat=lat_range)
    LON, LAT = np.meshgrid(da2.lon, da2.lat)
    ax.contourf(LON, LAT, da2, colors="k", transform=data_crs, 
                levels=levels_hatch, hatches=["", hatch_type], alpha=hatch_alpha)

# Plotting shaded countour, contour line and dotted figure
def hatch_contour_shaded(da_cs, da_cl, da_dot, lon_range, lat_range,
                         dlon=30, dlat=30, scale=0.7, title="",
                         cb_label="", cb_levels=21, cmap=plt.cm.RdBu_r, cb_pad=0.02, cb_width=20, cb_ticks=None,
                         cl_label=True, cl_levels=21, cl_linewidth=0.5, cl_labelfmt="%1.0f",
                         hatch_alpha=0, hatch_type="."):
    '''
    #Descriptions for parameters#
    da_cs: dataarray that you want to draw shaded contour. It must have 2D shape.
    da_cl: dataarray that you want to draw line contour. It must have 2D shape.
    lon_range, lat_range: select region that you want to draw.
    title: the title of this figure.
    dlon, dlat: set the gap of gridlines.
    scale: set the figure size.
    cb_label: the label of the colorbar.
    cl_label: if it is True, the label of line contour is added among the line contours. (default value: Ture)
    cb_levels: set the levels of shaded contours.
    cl_levels: set the levels of line contours.
    cmap: set the color map of shaded contour.
    cl_linewidth: set the line width of line contours.
    cl_labelfmt: set the format of line contour labels.
    cb_pad: the distance between the figure and the colorbar. it's proportional to the horizontal size of figure
    cb_width: the width of colorbar. when it is bigger, the width of color bar is smaller. (1/cb_width)
    cb_ticks: set the ticks of colorbar.
    hatch_alpha: set the transparency of hatches.
    hatch_type: default: '.', options: ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    '''
    # Rename instances
    da = da_cs
    da2 = da_cl
    levels = cb_levels
    levels2 = cl_levels
    # Sorting latitude into (-90 ~ 90)
    da = da.sortby(da.lat)
    da2 = da2.sortby(da2.lat)
    da_dot = da_dot.sortby(da_dot.lat)
    # Addcyclic
    lon_idx = da.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    da = xr.DataArray(wrap_data, coords=[da.lat, wrap_lon],
                      dims=["lat", "lon"])
    lon_idx = da2.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da2, coord=da2.lon, axis=lon_idx)
    da2 = xr.DataArray(wrap_data, coords=[da2.lat, wrap_lon],
                       dims=["lat", "lon"])
    lon_idx = da_dot.dims.index("lon")
    # longitude의 0도와 360도를 이어줌.
    wrap_data, wrap_lon = add_cyclic_point(da_dot, coord=da_dot.lon, axis=lon_idx)
    da_dot = xr.DataArray(wrap_data, coords=[da_dot.lat, wrap_lon],
                          dims=["lat", "lon"])

    # Shiftgrid
    if lon_range.start < 0:
        shiftgrid = True
    else:
        shiftgrid = False

    if shiftgrid:
        proj = ccrs.PlateCarree(central_longitude=0)
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)
        da2.coords["lon"] = (da2.coords["lon"] + 180) % 360 - 180
        da2 = da2.sortby(da2.lon)
        da_dot.coords["lon"] = (da_dot.coords["lon"] + 180) % 360 - 180
        da_dot = da_dot.sortby(da_dot.lon)
        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))
    else:
        proj = ccrs.PlateCarree(central_longitude=180)

        plt.figure(figsize=((lon_range.stop-lon_range.start)/10*(scale*1.2),
                            abs(lat_range.stop-lat_range.start)/10*scale))

    ax = plt.axes(projection=proj)
    ax.coastlines()  # parameters: 10, 50, 110m, default: 50m
    ax.gridlines(xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90, 91, dlat),
                 color="k", linestyle="dotted", linewidth=0.7)
    ax.set_xticks(np.arange(-180, 181, dlon), crs=proj)
    ax.set_yticks(np.arange(-90, 91, dlat), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=15)

    data_crs = ccrs.PlateCarree()
    da = da.sel(lon=lon_range, lat=lat_range)
    da2 = da2.sel(lon=lon_range, lat=lat_range)
    da_dot = da_dot.sel(lon=lon_range, lat=lat_range)

    # Draw a shaded figure
    LON, LAT = np.meshgrid(da.lon, da.lat)
    cs = ax.contourf(LON, LAT, da, transform=data_crs,
                     cmap=cmap, levels=levels, extend="both")
    cb = plt.colorbar(cs, orientation="vertical", pad=cb_pad, aspect=cb_width, ticks=cb_ticks)
    cb.set_label(cb_label)
    # Draw a contour line figure over the shaded figure
    LON, LAT = np.meshgrid(da2.lon, da2.lat)
    cl = ax.contour(LON, LAT, da2, transform=data_crs,
                    levels=levels2, colors="k", linewidths=cl_linewidth)
    if cl_label == True:
        plt.clabel(cl, inline=True, fmt=cl_labelfmt, fontsize=10, colors="k")
    # Draw a hatches figure for True/False dataset
    levels_hatch = [-0.5, 0.5, 1.5]  # True or False 분류를 위해
    LON, LAT = np.meshgrid(da_dot.lon, da_dot.lat)
    ax.contourf(LON, LAT, da_dot, colors="k", transform=data_crs, 
                levels=levels_hatch, hatches=["", hatch_type], alpha=hatch_alpha)


# Draw a rectangular box over the figure
def rect_box(box_area, edgecolor="k", linewidth=5):
    '''
    Note: this function should be used the next line of plotting figure code
    #Descriptions for parameters#
    box_area: [longitude range(list), latitude range(list)]
    '''
    box_lon_range = box_area[0]
    box_lat_range = box_area[1]

    LON1, LAT1 = (box_lon_range[0], box_lat_range[0])
    LON2, LAT2 = (box_lon_range[0], box_lat_range[1])
    LON3, LAT3 = (box_lon_range[1], box_lat_range[1])
    LON4, LAT4 = (box_lon_range[1], box_lat_range[0])
    poly = Polygon([(LON1, LAT1), (LON2, LAT2), (LON3, LAT3), (LON4, LAT4)],
                   facecolor="None", edgecolor=edgecolor, linewidth=linewidth)
    plt.gca().add_patch(poly)