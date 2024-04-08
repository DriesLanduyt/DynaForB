# -*- coding: utf-8 -*-
"""
Reference:
Johnson, I. (2013). DairyMod and the SGS Pasture Model: A mathematical description of the biophysical model structure

Created on Mon Apr 25 16:01:45 2016

@author: dlanduyt
"""
import math
import matplotlib.pyplot as plt
import csv
import numpy as np

##############
# Parameters #
##############

#solar constant omega (W/m²s)
omega = 1367
#atmospheric diffusivity constant
tau = 0.73
#PAR proportion of incoming irradiance
PARprop = 0.45

#############
# Functions #
#############

def Day_total(d,l,unit="PPF"):
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
    radiance = 3600*24*f*2/math.pi*tau*omega*math.sin(phi) #mean daily radiance J/m².d
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

##################
# Test functions #
##################
    
def radiation_curve(name,l,unit):
    x = range(365)
    y = [Day_total(i+1,l,unit) for i in x]
    plt.plot(x,np.array(y)[:,1],label=name)
    
def radiation_plot_layout(unit):
    plt.legend()
    plt.xlim([0,365])
    if unit=="RAD":plt.ylabel("Clear sky solar radiation (J/m².d)")
    elif unit=="PAR":plt.ylabel("Photosynthetic active radiation (J/m².d)")
    elif unit=="PPF":plt.ylabel("Photosynthetic photon flux (µmol photons/m².d)")
    plt.xlabel("Day of the year")
    
def test_radiation_plots(locationfile = "D:/Users/dlanduyt/Documents/PASTFORWARD/Model/Locations.csv",unit="PPF"):
    f = open(locationfile,'r')    
    locations = csv.reader(f)
    locations.__next__()
    for location in locations:
        radiation_curve(location[0],float(location[1]),unit)
    radiation_plot_layout(unit)
    f.close()