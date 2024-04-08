# -*- coding: utf-8 -*-
"""
Parameterfile DynaFORb

Created on Thu Apr  7 13:44:53 2016

@author: dlanduyt
"""

#PARAMETER DICTIONARY
#--------------------
Herbs = {
    #Herb species
    #------------

    "Species1":{
        "Description":"Anemone nemorosa",
        
        #species specific characteristics
        #--------------------------------
        #
        "Maturity_age": 10, #age when seed production starts
        #range: 0-15 year
        "SLA": 337.87, #specific leaf area [cm²/g]
        #range: 2-80 m²/kg or 20-800 cm²/g
        "SRL": 20, #specific root length [cm/g]
        #range: 10-500 m/g
        "Seed_mass": 1.96, #seed mass [mg/seed]
        #range: 10^{-3}-10^{7} mg
        "Max_seed_number": 50, #maximum number of seeds per plot for realistic bare ground initialization)
        
        "CNP": [80,10,1], #CNP ratio of plant biomass, should be MASS ratios
        #Range: [2000,20,1] - [200,10,1] (*based on leaf content)
        "WUE": 2.5, #photosynthetic or intrinsic water use efficiency [rate of carbon assimilation/rate of transpiration] [g C/mm H2O] of [g C/kg H20]
        
        "Strategy": "perennial", 
        
        "Min_mass":1,
        
        #PHOTOSYNTHESIS
        "k": 0.8,
        "Model_type":"NRH", #model used to estimate parameters (Exponential model (EM), Non-rectangular hyperbolic model (NRH),Rectangular hyperbolic model (RH))
        "Alpha":0.032, #Intrinsic quantum yield or radiation use efficiency [µmol.m−2.s−1 PPFD (photon flux density)]
        "Pnmax":13.94, #13.94 light saturated photosynthetic rate [µmol CO2.m-2.s-1]
        "Rd":0.768, #dark respiration rate
        "Angle":0.847, #curvilinear angle
        "Rm":0, #maintenance respiration [g C/g green biomass.d]
        "Rg":0, #growth respiration [proportion of gross photosynthesis]
            
        #RECRUITMENT
        "Germination_rate": 0.289, #percentage of seeds that germinate
        "Germination_trigger": 0, #expressed as degree_days (above 5°C)
        "Allocation_strategy_j": [0.4,0.2,0.4,0,0], #juvenile allocation strategy #shoot,leaves,root,seeds,reserves
        "Allocation_strategy_m": [0.3,0.1,0.3,0.25,0.05], #mature allocation strategy
        "Seedling_mass": [1,1,1,0,0], #dimensions that are reached after germination [g_shoot, g_leafs, g_roots, g_seeds, g_reserves]
        "Seedling_height":1, #height of a seedling in centimeters
        "Seedling_root_depth":1, #root depth of seedlings in centimeters
        
        #AGE-CLASS SPECIFIC
        "Max_height": [30]*10, #maximum height per age-class [cm]
        "Stress_tolerance": [4]*10, #stress tolerance of each age class [consecutive days]
        "Max_root_depth": [5]*10,
    },
    "Species2":{
        #species specific characteristics
        #--------------------------------
        "Description":"Species 2",
        "Maturity_age": 4, #age when seed production starts
        "SLA": 5, #specific leaf area [cm²/g]
        "SRL": 20, #specific root length [cm/g]
        "Seed_mass": 2, #seed mass [g/seed]
        "Max_seed_number": 10, #maximum number of seeds per plot for realistic bare ground initialization)
        "CNP": [80,10,1], #CNP ratio of plant biomass, should be MASS ratios
        "WUE": 2.5, #photosynthetic or intrinsic water use efficiency [rate of carbon assimilation/rate of transpiration] [g C/mm H2O] of [g C/kg H20]
        "Strategy": "perennial",
        "Min_mass":1,
            #photosysthesis
        "k": 0.8,
        "Model_type":"NRH", #model used to estimate parameters (Exponential model (EM), Non-rectangular hyperbolic model (NRH),Rectangular hyperbolic model (RH))
        "Alpha":0.032, #Intrinsic quantum yield or radiation use efficiency [µmol.m−2.s−1 PPFD (photon flux density)]
        "Pnmax":20, #light saturated photosynthetic rate [µmol CO2.m-2.s-1]
        "Rd":1.5, #dark respiration rate
        "Angle":0.847, #curvilinear angle
        "Rm":0.001, #maintenance respiration [mg C/mg green biomass.d]
        "Rg":0, #growth respiration [% of gross photosynthesis]
            #recruitment
        "Germination_rate": 0.5, #percentage of seeds that germinate
        "Germination_trigger": 100, #expressed as degree_days (above 5°C)
        "Allocation_strategy_j": [0.4,0.2,0.4,0,0], #juvenile allocation strategy #shoot,leafs,root,seeds,reserves
        "Allocation_strategy_m": [0.3,0.1,0.3,0.25,0.05], #mature allocation strategy
        "Seedling_mass": [1,1,1,0,0], #dimensions that are reached after germination [g_shoot, g_leafs, g_roots, g_seeds, g_reserves]
        "Seedling_height":2, #height of a seedling in centimeters
        "Seedling_root_depth":2, #root depth of seedlings in centimeters
        #age-class specific characteristics
        #----------------------------------
        "Max_height": [50]*4, #maximum height per age-class
        "Stress_tolerance": [4]*4, #stress tolerance of each age class [consecutive days]
        "Max_root_depth": [10]*4,
    }
}
    
Trees = {
    
    #Tree species
    #------------
    "QueRob":{
        "LAI_climax":10, #LAI of a mature forest ?dark phase when forest is young?
        "Growth_speed":1, #yearly increase of LAI from young to mature
        "Max_leaf_developmeny_speed":0.2,
        "Bud_break_trigger":50, #temperature or temperaturedays
        "k":0.8, #light attenuation factor
    },
    "FagSyl":{
        "LAI_climax":10, #LAI of a mature forest ?dark phase when forest is young?
        "Growth_speed":1, #yearly increase of LAI from young to mature
        "Max_leaf_developmeny_speed":0.2,
        "Bud_break_trigger":50, #temperature or temperaturedays
        "k":0.8, #light attenuation factor
    }
}

