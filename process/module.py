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
## Analysis functions
# Detrend along a single dimension
def detrend_poly(da, dim, deg=1):
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


# Correlation analysis for xarray.dataarray
def cov_xr(x, y, dims="time"):
    return xr.dot(x-x.mean(dims), y-y.mean(dims), dims=dims) / (x.count(dims) - 1)


def cor_xr(x, y, dims="time"):
    return cov_xr(x, y, dims) / (x.std(dims)*y.std(dims))


# Correlation analysis for numpy.ndarray: 두 data의 format이 다른 경우 사용
def cov_np(x, y):
    return np.dot(np.array(x)-np.array(x).mean(), np.array(y)-np.array(y).mean()) / (np.array(x).size-1)


def cor_np(x, y):
    return cov_np(x, y) / (np.array(x).std()*np.array(y).std())


# T-test for correlaiton coefficient (this test is also valid for linear regression coefficient)
def t_critical(N, sig_level=95.):
    q = (1 - 0.01 * sig_level) * 0.5
    t_critical = scipy.stats.t.ppf(q=1-q, df=N-2)
    return t_critical

def t_corr_stat_conv(N, da_corr):
    t = (da_corr*np.sqrt(N-2)) / np.sqrt(1-da_corr**2)
    return t

def t_test_corr(da_corr, time_size, sig_level=95.):
    t_crit = t_critical(N=time_size, sig_level=sig_level)
    right_tail = t_corr_stat_conv(N=time_size, da_corr=da_corr) >= t_crit
    left_tail  = t_corr_stat_conv(N=time_size, da_corr=da_corr) <= (-1) * t_crit
    return right_tail + left_tail

# 특정 array(time)에 대한 dataarray(time, lat, lon)의 (linear)regression analysis.
def regr(ar, da):
    n = ar.shape[0]
    result = da[0, :, :]
    a, b = np.polyfit(ar.values, da.values.reshape(n, -1), deg=1)
    a = a.reshape(result.shape)
    lon = da.lon
    lat = da.lat
    a = xr.DataArray(a, coords=[lat, lon], dims=["lat", "lon"])
    result = a
    return result

# Seasonal mean process for monthly datasets
def seasonal_mean(da, season):
    '''
    #Descriptions for parameters#
    da: dataarray for seasonal mean process. it should be monthly 3D(time, lat, lon) dataset
    season: the season you want to do the seaonal mean processing. parameters: "MAM", "JJA", "SON", "DJF"
    '''
    lat = da.lat
    lon = da.lon
    if season == "DJF":
        result = da[da.time.dt.season == season].resample(
            time="AS-Dec").mean(dim="time")
        time = pd.date_range(str(result.time.dt.year[0].values),
                             str(result.time.dt.year[-1].values+1), freq="Y").year
        result = xr.DataArray(result, coords=[time, lat, lon], dims=[
                              "time", "lat", "lon"])
        result = result.sel(time=slice(time[1], time[-2]))
    else:
        result = da[da.time.dt.season == season].resample(
            time="1Y").mean(dim="time")
        time = pd.date_range(str(result.time.dt.year[0].values),
                             str(result.time.dt.year[-1].values+1), freq="Y").year
        result = xr.DataArray(result, coords=[time, lat, lon], dims=[
                              "time", "lat", "lon"])
    return result


def sliding_std(arr, window=9):
    '''
    sliding standard deviation array 계산
    '''
    ls = []
    for i in range(arr.size-window+1):
        arr_trun = arr[i:window+i]
        ls.append(np.std(arr_trun))
    return ls


def sliding_dt_std(arr, window=9):
    '''
    sliding 후 detrend 처리, 
    그 뒤에 standard deviation 계산
    '''
    sigma = []
    for i in range(arr.size-window+1):
        arr_truncated = arr[i:window+i]
        trend = np.polyfit(np.arange(window), arr_truncated, 1)
        arr_trun_detrend = arr_truncated - trend[0]*np.arange(window) + trend[1]
        std = np.std(arr_trun_detrend)
        sigma.append(std)
    return sigma


def aave(da, box_range):
    '''
    Weighted area average
    #Descriptions for parameters#
    da: dataarray that you want to do area average process. it should be 3D(time, lat, lon) array.
    box_range: area that you wnat to do area average process. it should be the [lon_range, lat_range]. each lon_range and lat_range also be list format.
    '''
    lon_range = box_range[0]
    lat_range = box_range[1]
    da_box = da.sel(lon=slice(lon_range[0], lon_range[1]),
                    lat=slice(lat_range[0], lat_range[1]))
    weights = np.cos(np.deg2rad(da_box.lat))
    da_box_mean = da_box.weighted(weights).mean(dim=["lon", "lat"])
    return da_box_mean


# RSI algorithm
def rs_algorithm(arr, l=10, p=0.05):
    # Input data preprocessing
    # arr data type: xarray의 dataarray type or numpy의 ndarray type
    # 처리 과정 중에서는 np.ndarray format으로 다룸
    arr = np.array(arr)
    
    # Calculate diff.
    t     = scipy.stats.t.ppf(q=1-p/2, df=2*l-2)  # t critical value 계산
    std_l = np.array(sliding_std(arr, window=l)).mean()  # 전 기간에서 std의 평균값
    diff  = t*np.sqrt(2*(std_l**2)/l)
    
    # Calculate the values of initial regime
    x_r1_mean  = arr[:l].mean()
    up_thres   = x_r1_mean + diff
    down_thres = x_r1_mean - diff
    
    # Find a possible starting point of new regime 
    k = 0  # Initialize the number of regime shift occurrence
    RSI = {}  # Output initialization
    try:  # 맨 마지막 l개에서 indexerror 발생
        for i in range(arr.size-l):
            # 몇 번째 점인지 print out
            if (l+i+1) % 10 == 1:
                print(l+i+1, "st point:")
            elif (l+i+1) % 10 == 2:
                print(l+i+1, "nd point")
            elif (l+i+1) % 10 == 3:
                print(l+i+1, "rd point")
            else:
                print(l+i+1, "th point:")
                
            # RS algorithm core
            j = l + i
            if arr[j] > up_thres:  # possible starting point 선별조건. shift가 up일때
                print("Possible starting point")
                print("Shift is up")
                # Check RSI
                rsi = 0  # RSI summation value initialization
                # Possible starting point로 부터 l개의 점까지의 RSI summation
                for t in range(l):
                    x_ano = arr[j+t] - up_thres
                    rsi = rsi + x_ano/l/std_l
                    RSI[j] = rsi
                        
                    if rsi < 0:  # RSI summation이 음수라면 regime shift가 아님
                        print(f"Break at {t}th point\n")
                        RSI[j] = 0
                        break
                            
                    if t == l-1:  # l 번째 loop라면 작동. => RSI summation > 0
                        print("This point is the starting point of new regime")
                        k += 1
                        print("Let's find the next regime shift\n")
                        # Update mean value and thresholds to those of the next regime
                        x_rk_mean  = arr[j:j+l].mean()
                        up_thres   = x_rk_mean + diff
                        down_thres = x_rk_mean - diff
                            
            elif arr[j] < down_thres:  # possible starting point 선별조건. shift가 down일때
                print(arr[j], up_thres, down_thres)
                print("Shift is down")
                # Check RSI
                rsi = 0  # RSI summation value initialization
                # Possible starting point로 부터 l개의 점까지의 RSI summation
                for t in range(l):
                    x_ano = down_thres - arr[j+t]
                    rsi = rsi + x_ano/l/std_l
                    RSI[j] = rsi
                        
                    if rsi < 0:  # RSI가 음수라면 regime shift가 아님
                        print(f"Break at {t}th point\n")
                        RSI[j] = 0
                        break
                            
                    if t == l-1:  # l 번째 loop라면 작동. => RSI summation > 0
                        print("This point is the starting point of new regime")
                        k += 1
                        print("Let's find the next regime shift\n")
                        # Update mean value and thresholds to those of the next regime
                        x_rk_mean  = arr[j:j+l].mean()
                        up_thres   = x_rk_mean + diff
                        down_thres = x_rk_mean - diff

            # Update x_r1_mean (if k=0), up_thres, down_thres
            if k == 0:  # 아직 first regime일때
                # Update mean value and thresholds of the regime
                x_r1_mean  = arr[i+1:l+i+1].mean()
                up_thres   = x_r1_mean + diff
                down_thres = x_r1_mean - diff
                    
            print("This is not regime shift point")
            print("Go to the next point\n") 

    # Indexerror 발생 이유: 마지막 ㅣ-1개의 점들 중 possible starting point가 있다면 발생. for loop를 끝까지 못 실행하기 때문.    
    except IndexError:
        print("\nIndexerror occured, but it's ok")
        del RSI[list(RSI.keys())[-1]]
        
    return RSI, k