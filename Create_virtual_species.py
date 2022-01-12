# -*- coding: utf-8 -*-
"""
@author: dlanduyt
"""
import random as rd
import pandas as pd
import numpy as np

def Generate(spec,traits,n):
    
    #read trait database
    Herbs = pd.read_excel('data/Herbs.xlsx',skiprows=[0,2,3],index_col=0)
    
    #read upper and lower limit species
    ll_trait_array = Herbs.loc["Lower limit"]
    ul_trait_array = Herbs.loc["Upper limit"]

    #make empty dataframe with n copies of trait-array of existing species
    data = np.array([Herbs.loc["Anemone nemorosa"].values]*n)
    virt_spec_database = pd.DataFrame(data,columns = Herbs.columns)
    
    for i in range(n):
        virt_spec_database["Germination_rate"][i]=1
        virt_spec_database["Max_stress_tolerance"][i]='[365]'
        for t in traits: 
            virt_spec_database[t][i] = rd.uniform(ll_trait_array[t],ul_trait_array[t]) #when trad-offs are taken into account -> probability distribution conditional on all previously samples trait value
    
    return virt_spec_database


if __name__ == "__main__":
    
    #read trait database
    Herbs = pd.read_excel('data/Herbs.xlsx',skiprows=[0,2,3],index_col=0)
    
    #Generate virtual species
    virt_spec_data = Generate("Anemone nemorosa",['Alpha','Pnmax','Rd'],100)