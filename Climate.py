# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:29:32 2017

Based on E-OBS dataset, downloaded at 20/02/2017 
which contains data from the 1st of January 1950 till the 31th of August 2016 

@author: dlanduyt
"""
#locations
#home: 51.07; 3.76

from netCDF4 import Dataset,num2date
import numpy as np
import datetime as dt
import math
import pandas as pd

#global variables

#solar constant omega (W/m²s)
omega = 1367
#atmospheric diffusivity constant
tau = 0.73
#PAR proportion of incoming irradiance
PARprop = 0.45
#path to climate data
datapath = "E:/Users/dlanduyt/GIS/"

def Precipitation(lat,lon,dates):
    '''
    Units: mm/day
    '''
    return Get_EOBS_data('rr',lat,lon,dates)

def Temperature(lat,lon,dates): 
    '''
    Units: °C
    '''
    return Get_EOBS_data('tg',lat,lon,dates)/100
    
def Radiation(lat,lon,dates):
    '''
    Units: seconds and µmol/m².day
    '''
    return np.array([Day_Radiation(doy,lat) for doy in dates['dates'].dt.dayofyear])

def Get_EOBS_data(par,lat,lon,dates):
    """
    Extract data from E_OBS database v19.0e (data from 1/1/1950-31/12/2018)
    """
    
    #open file
    filename = "E-OBS/"+par+"_ens_mean_0.1deg_reg_v19.0e.nc"  #data from 1950-1-1 to 2018-31-1
    filepath = datapath + filename
    data = Dataset(filepath)
    data.set_auto_maskandscale(False)
    
    #extract coordinate series
    longitude = data.variables['longitude'][:]
    latitude = data.variables['latitude'][:]
    closest_lat_index = (np.abs(latitude-lat)).argmin()
    closest_lon_index = (np.abs(longitude-lon)).argmin()
    
    #define time indices to extract
    time = data.variables['time']
    time_in_dates = num2date(time[:],time.units,only_use_cftime_datetimes=False)
    dates_pd = pd.to_datetime(time_in_dates)

    mask = (dates_pd >= dates['dates'].iloc[0]) & (dates_pd <= dates['dates'].iloc[-1])
    
    #extract data based on mask and closest lat and lon
    output = np.array(data.variables[par][mask,closest_lat_index,closest_lon_index])
    
    data.close()
    
    return output
    
def Get_CRUTS_data(par,lat,lon,yb,ye):
    
    #open file
    filename="CRU TS 4.01/cru_ts4.01.1901.2016."+par+".dat.nc"
    filepath = datapath + filename
    data = Dataset(filepath,mode = "r")
    
    #extract coordinate series
    longitude = data.variables['lon'][:]
    latitude = data.variables['lat'][:]
    closest_lat_index = (np.abs(latitude-lat)).argmin()
    closest_lon_index = (np.abs(longitude-lon)).argmin()
    
    #define time indices to extract
    start_CRU = dt.datetime(1900,1,1) + dt.timedelta(days=int(data.variables['time'][0]))
    start_index = (yb-start_CRU.year)*12 
    stop_index =  (ye-start_CRU.year+1)*12 #stop_index points at first month of year after end year
    data = np.array(data.variables[par][start_index:stop_index,closest_lat_index,closest_lon_index])
    
    #deal with nodata entries
    
    
    return data
    
def Get_WorldClim_data(par,lat,lon,yb,ye):
    
    data = np.zeros(12)
    
    for m in range(1,13):
        
        month = str(m).zfill(2)
        
        filename= "WorldClim/"+par+"/wc2.0_10m_"+par+"_"+month+".tif"
        filepath = datapath + filename
        src = gdal.Open(filepath)
        
        #define pixel indices
        geo = src.GetGeoTransform()
        pixel_lon = int(float (lon - geo[0]) / abs(geo[1]))
        pixel_lat = int(float (geo[3] - lat) / abs(geo[5]))
        
        #get data for month
        data[m-1] = src.ReadAsArray()[pixel_lat,pixel_lon]*0.45 #account for proportion of PAR in total incoming radiation
        
    return np.array(list(data)*(ye-yb+1))

def Day_Radiation(d,l,unit="PPF"):
    """
    Computes radiation for a given day d (doy) at latitude l (degrees) in terms of photosynthetic photon flux (PPF) or Photosynthetic active radiation (PAR)
    """    
    #convert degrees in radians    
    l = l*math.pi/180
    
    #fraction of sun hours on day d    
    sigma = (math.pi/180)*23.45*math.sin(2*math.pi*((d-81)/365)) #solar declination angle
    f = (1/math.pi)*math.acos(-math.tan(l)*math.tan(sigma)) #daylength as a fraction of 24h

    #mean daily radiance
    phi = math.asin(math.sin(l)*math.sin(sigma)+math.cos(l)*math.cos(sigma)) #solar elevation angle at local noon
    radiance = 3600*24*f*(2/math.pi)*tau*omega*math.sin(phi) #mean daily radiance J/m².d
    par = PARprop*radiance #proportion of PAR in irradiance  
    
    if unit=="RAD":
        output = radiance
    elif unit=="PAR":
        output = par
    elif unit=="PPF":
        PPF = par/0.218  #personal communication Bruce Bugbee (ECOMOD documentation)
        output = PPF
    else: return "Unit not recognised"
    
    return [f*24*60*60,output] #[daylength[s],radiation]
    
    

    
    
    
    