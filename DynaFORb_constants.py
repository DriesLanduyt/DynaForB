# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:33:33 2016

@author: dlanduyt
"""
import numpy as np

constants = {
    "Molar_mass":{ #g mol
        "C":12.001,
        "N":14.007,
        "P":30.974 
    },
    "C_biomass_ratio":0.45, #mg C per mg biomass

#reference photosynthesis parameters
    "Pnmax_ref":{ #µmol/mol values extracted from Thornley and Johnson
        "C3":16,
        "C4":22
    },

#Pmax and alpha dependence on N mass content
    "N_ref":{ #mas % N values extracted from Thornley and Johnson
        "C3":0.04,
        "C4":0.03
    },

#Pmax dpendence on temperature
    "T_ref":{ #°C values extracted from Thornley and Johnson
        "C3":20,
        "C4":25
    },
    "T_min_pnmax":{ #°C values extracted from Thornley and Johnson
        "C3":3,
        "C4":12
    },
    "T_opt_pnmax":{ #°C values extracted from Thornley and Johnson
        "C3":23,
        "C4":35
    },
            
#Alpha dependence on temperature
    "T_opt_alpha":{ #°C values extracted from Thornley and Johnson
        "C3":15,
        "C4":np.nan
    },
    "T_lambda":{ #°C values extracted from Thornley and Johnson
        "C3":0.02,
        "C4":np.nan
    },
    "T_omega":{ #°C values extracted from Thornley and Johnson
        "C3":6,
        "C4":np.nan
    },
            
#maintenance respiration dependence on temperature
    
    "T_min_rm":{
        "C3":3,
        "C4":12
    },
            
#other constants
            
    "Survival_rate":0.5 #percentage that survives after stress level has passed maximum number of stress days
}

pars =  {
        "Dynamics":{
                "A":1,
                "B":0
                }}