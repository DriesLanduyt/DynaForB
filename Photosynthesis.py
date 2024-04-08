# -*- coding: utf-8 -*-
"""
Daily canopy photosynthesis, specified through the following parameters:

I      PAR photon flux density [µmol photons.m-2.s-1]
Pnmax  light saturated photosynthetic rate [µmol CO2.m-2.s-1]
alpha  Intrinsic quantum yield or radiation use efficiency [µmol CO2.mol photosynthetic flux density-1]
Rd     Dark respiration
angle  dimensionless angle of curvature

Reference: 

Note: 
depending on the definition and units of alpha, gross photosynthesis is expressed as mg C02/s.m² leaf or 

Created on Wed Jul 27 09:57:15 2016

@author: dlanduyt
"""

import math
import numpy as np
from DynaFORb_constants import constants

def Net_gross_photosynthesis(LA_within_layer,LAI_above_layer,light,k,p_pars):
    """
    Calculates daily carbon gain for one specific vegetation layer [gC/m² soil.day]
    """
    I = light[1]*math.e**(-LAI_above_layer) #transmitted light (k as an empirical coefficient (combination of leaf inclination and light transmittance))
    m = 0 #no light transmittance assumption
    I = (k/(1-m))*I  #light incident on leafs
    daylength = light[0]
    I_mean = I/daylength 
    P = LA_within_layer*[globals()[p_pars[0,i]](I_mean[i],float(p_pars[1,i]),float(p_pars[2,i]),float(p_pars[3,i]),float(p_pars[4,i])) for i in range(len(LA_within_layer))]
    P_day = daylength*P #[µmol/m².day]
    C_inc = P_day*constants["Molar_mass"]["C"]*10**-6 #conversion from µmol to mol (*10^-6) and from mol CO2 to g C (*12)
    C_inc[C_inc<0]=0
    return C_inc
        
def NRH(I,Pnmax,alpha,Rd,angle):
    """
    Non-rectangular hyperbolic model [µmol C/m² leaf.s]
    """
    Pn = (alpha*I+Pnmax-math.sqrt((alpha*I+Pnmax)**2-4*I*alpha*angle*Pnmax))/(2*angle)-Rd #[µmol/m².s]
    
    return Pn

def RH(I,Pnmax,alpha,Rd,angle):
    """
    Rectangular hyperbolic model [µmol/m² leaf.s]
    """
    Pn = alpha*I*Pnmax/(alpha*I+Pnmax)-Rd #[µmol/m².s]
    
    return Pn
    
def EM(I,Pnmax,alpha,Rd,angle):
    """
    Exponential model
    """
    Pn = Pnmax*(1-math.exp(-alpha*I/Pnmax))-Rd
    Pn = Pn/10000 #[µmol.cm-2.s-1]
    
    return Pn
    
    
def FBvC(I,T,Vcmax):
    
    #Parameters and equations derived from Pury and Farquhar (1997)
    #Constants: 
    Kc = 40.4 #Pa
    Ko = 24.8*10**3 #Pa    
    LCP = 3.69
    Jmax = 2.1*Vcmax
    Rd = 0.0089*Vcmax
    Theta = 0.7
    f = 0.15
    R = 8.314
    H = 220000 #J/mol
    S = 710 #J/K*mol
    
    #Intercellular CO2 and O2 concentrations
    Ci = 24.5 #generally modelled with empirical stomata conductance model
    Oi = 20.5*10**3 #Pa
    
    #effect of temperature on model parameters Jmax, Kc, Ko and LCP 
    Tk = T+273.15
    Ea = 37000
    Jmax = Jmax*math.exp(((Tk-298)*Ea)/(R*Tk*298))*(1+math.exp((S*298-H)/(R*298)))/(1+math.exp((S*Tk-H)/(R*Tk))) 
    
    Ea = 59400
    Kc =  Kc*math.exp((Ea*(T-25))/(298*R*(T+273))) #Arrhenius function
    
    Ea = 36000
    Ko = Ko*math.exp((Ea*(T-25))/(298*R*(T+273)))
    
    LCP = LCP + 0.188*(T-25)+0.0036*(T-25)**2
    
    #rubisco-limited photosynthesis
    Av = Vcmax*(Ci-LCP)/(Ci+Kc*(1+Oi/Ko))
    
    #Electron-transport-limited photosynthesis
    #
    #J as solution of Theta*J²-(0.5*(1-f)*I+Jmax)*J+0.5*(1-f)*I*Jmax = 0
    a = Theta
    b = -(Jmax+0.5*(1-f)*I)
    c = 0.5*(1-f)*I*Jmax #0.5*(1-f) reflects alpha in NRH
    J1 = (-b + math.sqrt(b**2 - 4 * a * c) ) / (2 * a)
    J2 = (-b - math.sqrt(b**2 - 4 * a * c) ) / (2 * a)
    J =  min(J1, J2)
    #Assimilation if limited by electron transport
    Aj = J*(Ci-LCP/(4*(Ci+2*LCP)))
    
    return [Aj,Av,min(Aj,Av)-Rd]