# -*- coding: utf-8 -*-
"""
Parameterfile DynaFORb

Created on Thu Apr  7 13:44:53 2016

@author: dlanduyt
"""
import numpy as np

#PARAMETER DICTIONARY
#--------------------
def Layers(number_of_layers,upper_limit):
    resolution = upper_limit/number_of_layers
    return np.arange(resolution,upper_limit+resolution,resolution)


par={
    #plot characteristics
    "Position":[50.97,3.80],#Aalmoeseneiebos
    "Plot_size":100,  # [cm²] should be 100 to 1000 times the surface of a single leaf to stay below 0.05 relative error while calculating light attenuation   
    "Bulk_density":1.1, # [ton/m³]
    "Tree_species":"QueRob",
    "Tree_density": 0.8,
    "Tree_age":2,
    #resolution
    "Vegetation_layers":Layers(100,50),
    "Soil_layers":Layers(1,10), 
    #initial conditions
    "PFTs":["Species1","Species2"],
    "Initial_vegetation":"Test", #can be "Bare","Test"
    "Initial_soil_resources":[[10,1.1,10]], #seperated over different layers (2D array with shape nx3) or one value for each resource (1D array with 3 elements)
    #scenarios    
    "Climate_change":0,
    "Meteo_data":"Data/Eindhoven_2015.csv"
    }

#can include more variants of the par dictionary depending on the case studies under investigation

