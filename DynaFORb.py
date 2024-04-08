# -*- coding: utf-8 -*-
"""

-------------------------------------------------------------------------------------
DynaFORb - a process-based model to describe herb layer dynamics in temperate forests
-------------------------------------------------------------------------------------   

---------------------------------------
Author:     Dries Landuyt
Address:    Geraardsbergsesteenweg 267
            9090 Melle (Gontrode) 
            Belgium
Contact:    dries.landuyt@ugent.be
Project:    ERC project PASTFORWARD
---------------------------------------

"""

#switch process
light_comp = 'yes' #yes or no

#Import external python packages
import pandas as pd
import numpy as np
import math as m

#Import internal python packages
import Climate
from DynaFORb_constants import constants,pars
from Create_virtual_species import Generate
from Photosynthesis import Net_gross_photosynthesis


#Load datafiles with trait data
Herbs = pd.read_excel('data/Herbs.xlsx',skiprows=[0,2,3],index_col=0)
Trees = pd.read_excel('data/Trees.xlsx',skiprows=[1,2,3],index_col=0)
PhytoCalc = pd.read_excel('data/PhytoCalc.xlsx',index_col=0)

#tests
test_seeds_1 = 0
test_seeds_2 = 5
    
class Plot:
    
    def __init__(self, period=['1/1/1950','10/12/1951'], loc=[51,4], size=200, PFTs = Herbs,
                 h_spec=["Anemone nemorosa","Urtica dioica"],
                 s_layers=[1,20], s_res_t0=[10000,1.1,10],
                 veg_layers=[10,50], veg_cover_t0=[0.1,0.1], veg_height_t0=[0,0], 
                 t_spec=["QueRob"],t_comp=[1], LAI_min=0, LAI_max=5,
                 macroclimate_data = ["","","",""], #[lightdatafile,daylengthdatafile,temperaturedatafile,precipitationdatafile]
                 microclimate_data = ["","","",""], #[lightdatafile,daylengthdatafile,temperaturedatafile,precipitationdatafile]
                 microclimate_treatment = [0,0], #[lighttreatment during daylight hours (µmol/m².s),temperaturetreatment in increase in daily mean temperature (°C)]
                 soil_treatment = [0,0,0] #[Ntreatment,Ptreatment,watertreatment]
                 ):
        
        """
        Initiate plot with position tuple (latitude,longitude), plot size [cm²], 
        resolution of soil and vegetation layers and initial soil, vegetation 
        and canopy conditions (N,P,water), period date tuple is formatted mm/dd/yyyy, both dates are included
        """
        
        #Plot characteristics
        self.size = size
        
        #Initialize model components
        self.dates = pd.date_range(start=period[0],end=period[1]) #includes last date or, if last date is a year (e.g. period = ['2017','2018']), includes the 1st of January of last date/year
        self.drivers = Drivers(loc,self.dates,macroclimate_data)
        self.canopy = Canopy(t_spec,t_comp,LAI_min,LAI_max,self.dates)
        self.microclimate = Microclimate(self.dates,microclimate_data,microclimate_treatment)
        self.soil = Soil(s_layers,s_res_t0,soil_treatment)
        self.vegetation = Vegetation(PFTs,h_spec,veg_cover_t0,veg_height_t0,veg_layers,s_layers,self.size)
        
    def Adjust_vegetation(self,PFTs,h_spec,veg_cover_t0,veg_height_t0,veg_layers,s_layers):
        """
        Adjust a component of the model
        """
        self.vegetation = Vegetation(PFTs,h_spec,veg_cover_t0,veg_height_t0,veg_layers,s_layers,self.size)
            
    def Simulate(self):
        
        """
        Scheduler 
        """
        
        #initialize output dataframe
        output_df = pd.DataFrame()
        #define index variables
        output_df['dates'] = np.repeat(self.dates,len(self.vegetation.species))
        output_df['PFT'] = np.tile(self.vegetation.species,len(self.dates))
        #initialize output variables
        output_df['height'] = None
        output_df['cover'] = None
        output_df['photosynthesis'] = None
        output_df['performance'] = None
        output_df['limiting_resource'] = None
        
        
        dates_df = pd.DataFrame({'dates':self.dates})
        
        #Start simulation loop
        for n,year_dates in dates_df.groupby(pd.Grouper(key='dates',freq='Y')):
            
            #display simulation year and progress in %
            #print('Year '+str(n.year))
            
            #within year dynamics of external drivers
            self.drivers.Dynamics(year_dates) #sets drivers as subsets of the long-term daily driver data ->> convert into dataframe that can be used as argument for microclimate calculations
            self.canopy.Dynamics(self.drivers.dynamics) #uses drivers of above, mainly degreedays
            self.microclimate.Dynamics(self.drivers.dynamics,self.canopy.dynamics) #uses canopy and driver dynamics
            self.soil.Dynamics(self.canopy.dynamics,self.microclimate.dynamics) #uses drivers but also canopy conditions
            self.vegetation.Dynamics(self.microclimate.dynamics) #sets self.vegetation.height: height per day per PFT based on the given degreedays
            
            #soil and microclimate define resources and conditions for understorey growth
            growth_inputs = self.soil.dynamics.merge(self.microclimate.dynamics,on='dates').merge(self.vegetation.dynamics,on='dates')
            
            
            #calculate understorey performance #TO DO from here onwards
            growth_result = np.array([self.vegetation.Grow(res_and_cond) for index,res_and_cond in growth_inputs.iterrows()])
            photosynthesis = growth_result[:,0,:]
            performance = growth_result[:,1,:] #all days,only performance,all PFTs
            limiting_resource = growth_result[:,2,:]  #all days,only performance,all PFTs
            
            #add result to output data
            #add growth conditions to output array    
            output_df.loc[output_df['dates'].dt.year==n.year,'height'] = np.array(self.vegetation.dynamics.loc[:, self.vegetation.dynamics.columns.str.startswith('PFT')]).flatten()
            output_df.loc[output_df['dates'].dt.year==n.year,'cover'] = np.tile(self.vegetation.cover,len(year_dates))
            output_df.loc[output_df['dates'].dt.year==n.year,'photosynthesis'] = photosynthesis.flatten()
            output_df.loc[output_df['dates'].dt.year==n.year,'performance'] = performance.flatten()
            output_df.loc[output_df['dates'].dt.year==n.year,'limiting_resource'] = limiting_resource.flatten()  

            #update cover based on performance and age vegetation
            year_performance = np.array([np.mean(r[r>0]) if np.any(r>0) else 0 for r in performance.transpose()]) #mean over days for all positive values
            self.vegetation.Update(year_performance)
            
            #shift cover values within species to higher age classes, if available
            #self.vegetation.Age()
        
        #add driver dynamics to output
        output_df = output_df.merge(self.microclimate.regime,how='left',left_on='dates',right_on='dates')
        
        return output_df
            
class Drivers():
    
    def __init__(self,loc,period,data):
        """
        Loads the environmental drivers data for the entire simulation period
        """
        
        if len(loc)!=2:
            raise DynaFORbError('loc argument should be a list with two values: [latitude,longitude]')
        
        #initialize regime dataframe
        self.regime = pd.DataFrame()
        self.regime['dates'] = period
        
        dates = pd.DataFrame({'dates':period},index=period)
        
        #load drivers for the simulation period
        #--------------------------------------
        
        #Load or model PAR irradiation
        if data[0]=="":
            radiationdata = pd.DataFrame({'light':Climate.Radiation(loc[0],loc[1],self.regime)[:,1]},index=period)
        else:
            radiationdata = pd.read_csv(data[0],index_col=0,parse_dates=True)
            radiationdata = radiationdata[radiationdata.index==period]
            radiationdata.columns = ['light']
        
        #Load or model daylength data
        if data[1]=="":
            daylengthdata = pd.DataFrame({'daylength':Climate.Radiation(loc[0],loc[1],self.regime)[:,0]},index=period)
        else:
            daylengthdata = pd.read_csv(data[1],index_col=0,parse_dates=True)
            daylengthdata = daylengthdata[daylengthdata.index==period]
            daylengthdata.columns = ['daylength']
            
        #Load or model temperature data
        if data[2]=="":
            temperaturedata = pd.DataFrame({'temperature':Climate.Temperature(loc[0],loc[1],self.regime)},index=period)
        else:
            temperaturedata = pd.read_csv(data[2],index_col=0,parse_dates=True)
            temperaturedata = temperaturedata[temperaturedata.index==period]
            temperaturedata.columns = ['temperature']
            
        #Load or model precipitation data
        if data[3]=="":
            precipitationdata = pd.DataFrame({'precipitation':Climate.Precipitation(loc[0],loc[1],self.regime)},index=period)
        else:
            precipitationdata = pd.read_csv(data[3],index_col=0,parse_dates=True)
            precipitationdata = precipitationdata[precipitationdata.index==period]
            precipitationdata.columns = ['precipitation']

#        #Load or model N deposition data
#        if data[4]=="":
#            Ndepdata = 
#        else:
#            Ndepdata = pd.read.csv(data[4])
#        self.regime['N_deposition'] = Ndepdata
        
        #merge all drivers
        self.regime = pd.concat([dates,radiationdata,daylengthdata,temperaturedata,precipitationdata],axis=1)
        
        #calculate degreedays
        temp_adjusted = self.regime['temperature']-5
        temp_adjusted[temp_adjusted<0]=0
        self.regime['degreedays'] = np.cumsum(temp_adjusted)
        
    def Dynamics(self,year_dates):
        """
        Takes a subset of the environmental drivers data for the year that is simulated
        """
        #initialize dynamics variable
        self.dynamics = self.regime.loc[year_dates.dates]
        
#        #take a subset for the specified year
#        mask = (self.regime['dates']>=year_dates['dates'].iloc[0]) & (self.regime['dates']<=year_dates['dates'].iloc[-1])
#        self.dynamics = self.regime.loc[mask]
#        self.dynamics = self.dynamics.reset_index(drop=True) 
        
class Canopy():
    
    def __init__(self,t_spec,t_comp,LAI_min,LAI_max,period):
        """
        Generates a canopy with a specific species, composition of species and a modelled LAI trend
        """
        
        #check input
        if type(t_spec) is not list:
            raise DynaFORbError('t_spec argument should be a list with all tree species in the canopy')
        if type(t_comp) is not list:
            raise DynaFORbError('t_comp argument should be a list with the cover values for all tree species in the canopy')
        if len(t_spec) != len(t_comp):
            raise DynaFORbError('Provided number of tree species does not match with number of provided cover values')
        
        #set parameters
        self.species = t_spec
#        self.bud_burst = np.sum(np.array([Trees["dd_bud_burst"][s] for s in self.species])*np.array(t_comp))
#        self.leaf_fall = np.sum(np.array([Trees["dd_leaf_fall"][s] for s in self.species])*np.array(t_comp))
        self.k = np.sum(np.array([Trees["k"][s] for s in self.species])*np.array(t_comp))
        self.TdayS = np.mean([Trees['TdayS'][s] for s in self.species]) #not yet optimal, best to calculate dynamics per species and then sum up their LAI
        self.TdayF = np.mean([Trees['TdayF'][s] for s in self.species])
        self.cT = np.mean([Trees['Max_leaf_development_speed'][s] for s in self.species])
        
        #define regime of LAI_min and LAI_max (trend over the years)
        self.regime = pd.DataFrame()
        self.regime['year'] = np.unique(period.year)
        try: self.regime['LAI_min'] = LAI_min #can take both one value and a series of values (one for each simulated year)
        except: raise DynaFORbError('Provided number of LAI_min values do not match number of simulated years') 
        try: self.regime['LAI_max'] = LAI_max #idem
        except: raise DynaFORbError('Provided number of LAI_max values do not match number of simulated years')
    
    def Dynamics(self,driver_dynamics):
        """
        Calculates overstorey phenology
        """
        #initialize dynamics dataframe
        self.dynamics = pd.DataFrame()
        
        #look up min and max LAI for modelled year
        year = np.unique(driver_dynamics['dates'].dt.year)[0]
        LAI_min = float(self.regime['LAI_min'][self.regime['year']==year])
        LAI_max = float(self.regime['LAI_max'][self.regime['year']==year])
        
        #determine dynamics based on climate (see also Niinemets et al., 2005 and Xie et al., 2005 for more complex representations)
        self.dynamics['dates']  = driver_dynamics['dates']
        self.dynamics['LAI_up'] = (-1/self.k)*np.log(np.power(m.e,-self.k*LAI_max)+(np.power(m.e,-self.k*LAI_min)-np.power(m.e,-self.k*LAI_max))/(1+np.power(m.e,self.cT*(np.array(self.dynamics['dates'].dt.dayofyear)-self.TdayS))))
        self.dynamics['LAI_down'] = (-1/self.k)*np.log(np.power(m.e,-self.k*LAI_min)-(np.power(m.e,-self.k*LAI_min)-np.power(m.e,-self.k*LAI_max))/(1+np.power(m.e,self.cT*(np.array(self.dynamics['dates'].dt.dayofyear)-self.TdayF))))
        self.dynamics['LAI'] = self.dynamics['LAI_up']
        self.dynamics.loc[self.dynamics['dates'].dt.dayofyear>182,'LAI'] = self.dynamics.loc[self.dynamics['dates'].dt.dayofyear>182,'LAI_down']  
#        self.dynamics['LAI'] = LAI_min
#        
#        self.dynamics.loc[driver_dynamics['degreedays']>self.bud_burst,'LAI'] = LAI_max
#        self.dynamics.loc[driver_dynamics['degreedays']>self.leaf_fall,'LAI'] = LAI_min
        self.dynamics['k'] = self.k
        
class Microclimate():
    
    def __init__(self,period,data,microclimate_treatment):
        """
        Initialize microclimate
        """
        #define period
        dates = pd.DataFrame({'dates':period},index=period)
        
        #load microclimate data
        if data[0]=="":
            radiationdata = pd.DataFrame({'light':np.nan},index=period)
        else:
            radiationdata = pd.read_csv(data[0],index_col=0,parse_dates=True)
            radiationdata = radiationdata[radiationdata.index==period]
            radiationdata.columns = ['light']
            
        if data[1]=="":
            daylengthdata = pd.DataFrame({'daylength':np.nan},index=period)
        else:
            daylengthdata = pd.read_csv(data[1],index_col=0,parse_dates=True)
            daylengthdata = daylengthdata[daylengthdata.index==period]
            daylengthdata.columns = ['daylength']
            
        if data[2]=="":
            temperaturedata = pd.DataFrame({'temperature':np.nan},index=period)
        else:
            temperaturedata = pd.read_csv(data[2],index_col=0,parse_dates=True)
            temperaturedata = temperaturedata[temperaturedata.index==period]
            temperaturedata.columns = ['temperature']
            
        if data[3]=="":
            precipitationdata = pd.DataFrame({'precipitation':np.nan},index=period)
        else:
            precipitationdata = pd.read.csv(data[3],index_col=0,parse_dates=True)
            precipitationdata = precipitationdata[precipitationdata.index==period]
            precipitationdata.columns = ['precipitation']
            
        self.regime = pd.concat([dates,radiationdata,daylengthdata,temperaturedata,precipitationdata],axis=1)
        
        #calculate degreedays
        temp_adjusted = self.regime['temperature']-5
        temp_adjusted[temp_adjusted<0]=0
        self.regime['degreedays'] = np.cumsum(temp_adjusted)
        
        self.treatment = microclimate_treatment
        
    def Dynamics(self,driver_dynamics,canopy_dynamics):
        """
        Converts all drivers into microclimatic drivers, experienced by the understorey
        """
        
        #load microclimate data
        self.dynamics = self.regime.loc[driver_dynamics.dates]
        
        #model microclimate if no data available
        if self.dynamics['light'].isnull().all():
            self.dynamics['light'] = driver_dynamics["light"]*np.exp(-canopy_dynamics['k']*canopy_dynamics['LAI']) + self.treatment[0]*driver_dynamics['daylength']
    
        if self.dynamics['daylength'].isnull().all():
            self.dynamics["daylength"] = driver_dynamics["daylength"]

        if self.dynamics["temperature"].isnull().all():
            self.dynamics["temperature"] = driver_dynamics["temperature"] + self.treatment[1] #potentially adjust based on LAI?
        
        if self.dynamics["precipitation"].isnull().all():
            self.dynamics["precipitation"] = driver_dynamics["precipitation"] #potentially adjust based on LAI?
        
        #calculate degreedays
        temp_adjusted = self.dynamics['temperature']-5
        temp_adjusted[temp_adjusted<0]=0
        self.dynamics["degreedays"] = np.cumsum(temp_adjusted)
        
        #update microclimate regime
        self.regime.loc[driver_dynamics.dates] = self.dynamics.loc[driver_dynamics.dates]

class Vegetation():

    def __init__(self,PFTs,h_spec,veg_cover_t0,veg_height_t0,shoot_layers,root_layers,size):
        """
        Generates a community matrix for the provided PFTs and according to the provided method ("Bare","Random_seed_rain")
        """
        
        #check input
        if type(PFTs) is not pd.DataFrame:
            raise DynaFORbError('PFTs argument should be a pandas dataframe with plant traits as columns and species as rows')
        if type(h_spec) is not list:
            raise DynaFORbError('h_spec argument should be a list with names of all herb species being modelled')
        if len(shoot_layers)!=2: 
            raise DynaFORbError('vegetation layer argument only accepts an array with two value: [number of layers,total height to be divided in layers]')
        if len(root_layers)!=2: 
            raise DynaFORbError('soil layer argument only accepts an array with two value: [number of layers,total depth to be divided in layers]') 
        
        #set species composition
        self.unique_species = h_spec
        self.species = np.repeat(self.unique_species,[PFTs["Maturity_age"][s] for s in self.unique_species])
        
        #further check of input
        if len(veg_cover_t0)!=len(self.species):
            raise DynaFORbError("Provided initial cover values do not match with number of species and/or age classes to be modelled")
        if len(veg_height_t0) !=len(self.species):
            raise DynaFORbError("Provided initial height values do not match with number of species and/or age classes to be modelled")
        
        #set resolution parameters
        self.size = size
        self.shoot_layers = Layers(shoot_layers[0],shoot_layers[1])
        self.root_layers =  Layers(root_layers[0],root_layers[1])       
        
        #masks to denote locations of mature plants and one-year-olds within vegetation arrays        
        self.age = np.concatenate([np.arange(1,PFTs["Maturity_age"][s]+1) for s in self.unique_species],0)
        self.seedling = self.age==1
        self.mature = np.roll(self.seedling,-1)
        
        #load general characteristics
        self.strategy = np.array([PFTs["Strategy"][s] for s in self.species])
        
        #competition characteristics to full array
        self.sla = np.array([PFTs["SLA"][s] for s in self.species])
        self.srl = np.array([PFTs["SRL"][s] for s in self.species])
        self.max_height = np.array([PFTs['Max_height'][s] for s in self.species])
        self.root_depth = np.array([PFTs['Root_depth'][s] for s in self.species])
        self.WUE = np.array([PFTs["WUE"][s] for s in self.species])
        self.SR = np.array([PFTs["Shoot_root_ratio"][s] for s in self.species])
        LF = np.array([PFTs["Leaf_fraction"][s] for s in self.species])
        leaf_fraction = (self.SR/(1+self.SR))*LF #leaf fraction of total plant biomass [/]
        root_fraction = (1/(self.SR+1)) #root fraction of total plant biomass [/]
        self.architecture = np.vstack((leaf_fraction,root_fraction)).transpose()
        self.LNC = np.array([PFTs["LNC"][s] for s in self.species])
        self.LPC = np.array([PFTs["LPC"][s] for s in self.species])
            # photosynthesis parameters        
        self.pathway = np.array([PFTs["Pathway"][s] for s in self.species])
            # Pnmax and regulators
        self.pnmax = np.array([PFTs["Pnmax"][s] for s in self.species])
        self.Nref = np.array([constants["N_ref"][PFTs["Pathway"][s]] for s in self.species])
        self.Tref = np.array([constants["T_ref"][PFTs["Pathway"][s]] for s in self.species])
        self.Tmin_pnmax = np.array([constants["T_min_pnmax"][PFTs["Pathway"][s]] for s in self.species])
        self.Topt_pnmax = np.array([constants["T_opt_pnmax"][PFTs["Pathway"][s]] for s in self.species])
            # Alpha and regulators
        self.alpha = np.array([PFTs["Alpha"][s] for s in self.species])
        self.Topt_alpha = np.array([constants["T_opt_alpha"][PFTs["Pathway"][s]] for s in self.species])
        self.lambdaT = np.array([constants["T_lambda"][PFTs["Pathway"][s]] for s in self.species])
        self.omegaT = np.array([constants["T_omega"][PFTs["Pathway"][s]] for s in self.species])
            # Rd and regulators
        self.rd = np.array([PFTs["Rd"][s] for s in self.species])
        self.angle = np.array([PFTs["Angle"][s] for s in self.species])
        self.p_model = np.array([PFTs["Model_type"][s] for s in self.species])
        self.k = np.array([PFTs["k"][s] for s in self.species])
            # Respiration and regulators
        self.rm = np.array([PFTs["Rm"][s] for s in self.species])
        self.Tmin_rm = np.array([constants["T_min_rm"][PFTs["Pathway"][s]] for s in self.species])
        self.rg = np.array([PFTs["Rg"][s] for s in self.species])
            # phenology parameters
        self.start = np.array([PFTs["Start"][s] for s in self.species])
        self.max1 = np.array([PFTs["Max1"][s] for s in self.species])
        self.max2 = np.array([PFTs["Max2"][s] for s in self.species])
        self.stop = np.array([PFTs["Stop"][s] for s in self.species])
            # immigration parameters
        self.seedmass = np.array([PFTs["Seed_mass"][s] for s in self.unique_species])
        
        # Set allometric parameters
        classes = [PFTs['PhytoCalc_class'][s] for s in self.species]
        self.a = np.array([PhytoCalc["a"][c] for c in classes])
        self.b = np.array([PhytoCalc["b"][c] for c in classes])
        self.c = np.array([PhytoCalc["c"][c] for c in classes])
        
        #initialize state variables
        self.cover = np.array(veg_cover_t0) #[%]
        self.height = np.array(veg_height_t0) #[cm]
        self.biomass = self.size*(1+1/self.SR)*(self.a*self.cover**self.b*self.height**self.c) #total above and belowground biomass [g/m²]

    
    def Dynamics(self,microclimate_dynamics):
        """
        Calculates understorey phenology
        """
        #initialize dynamics dataframe
        self.dynamics = pd.DataFrame()
        self.dynamics['dates'] = np.array(microclimate_dynamics['dates'])
        
        #automatic switch between two approaches: degreeday based or based on months 
        if all(np.isin(self.max1,np.arange(1,13))):
            start_indices = [self.dynamics.index[self.dynamics.dates.dt.month==T].min() - 15 for T in self.max1] #assuming emergence 15 days prior to first flowering date
            max1_indices = [self.dynamics.index[self.dynamics.dates.dt.month==T].min() for T in self.max1]
            max2_indices = [self.dynamics.index[self.dynamics.dates.dt.month==T].min() for T in self.max2]
            stop_indices = [self.dynamics.index[self.dynamics.dates.dt.month==T].min() for T in self.max2] #assuming senescence immediately after flowering
        else:
              start_indices = [max(np.argmax(np.array(microclimate_dynamics["degreedays"])>t) - 15,0) for t in self.max1] #assuming emergence 15 days prior to first flowering date
              max1_indices = [np.argmax(np.array(microclimate_dynamics["degreedays"])>t) for t in self.max1]
              max2_indices = [np.argmax(np.array(microclimate_dynamics["degreedays"])>t) for t in self.max2]
              stop_indices = [np.argmax(np.array(microclimate_dynamics["degreedays"])>t) for t in self.max2] #assuming senescence immediately after flowering
      
        for i in range(len(self.species)):
            colname = 'PFT'+str(i)
            self.dynamics[colname]= 0
            if self.cover[i]!=0:
                if self.strategy[i] == 'D':
                    self.dynamics.loc[start_indices[i]:max1_indices[i]-1,colname] = self.max_height[i]*(self.dynamics.index[start_indices[i]:max1_indices[i]]-start_indices[i])/(max1_indices[i]-start_indices[i])
                    self.dynamics.loc[max1_indices[i]:max2_indices[i]-1,colname] = self.max_height[i]
                    self.dynamics.loc[max2_indices[i]:stop_indices[i]-1,colname] = self.max_height[i]*(1- (self.dynamics.index[max2_indices[i]:stop_indices[i]]-max2_indices[i])/(stop_indices[i]-max2_indices[i]))
                    self.dynamics.loc[stop_indices[i]:,colname] = 0
                else:
                    self.dynamics.loc[:,colname] = self.max_height[i]
            
    def Grow(self,growth_input):
        
        """
        Calculate daily growth of the plot's vegetation and soil resource uptake
        """
        
        #ABOVEGROUND ASYMMETRIC COMPETITION
        #----------------------------------
        
        #Adapt photosynthesis and respiration parameters to contemporary conditions from growth_input dataframe
        T= growth_input['temperature']
        #pnmax
        q=2.5
        Tmax=((1+q)*self.Topt_pnmax-self.Tmin_pnmax)/q
        pnmax_T_fac = np.zeros(len(Tmax)) #zero for very low and high T values
        con =  (T>self.Tmin_pnmax) & (T<Tmax)
        pnmax_T_fac[con] = (((T-self.Tmin_pnmax[con])/(self.Tref[con]-self.Tmin_pnmax[con]))**q)*(((1+q)*self.Topt_pnmax[con]-self.Tmin_pnmax[con]-q*T)/((1+q)*self.Topt_pnmax[con]-self.Tmin_pnmax[con]-q*self.Tref[con]))
        pnmax_T_fac[(T>self.Topt_pnmax) & (self.pathway=="C4")]= (((self.Topt_pnmax-self.Tmin_pnmax)/(self.Tref-self.Tmin_pnmax))**q*(((1+q)*self.Topt_pnmax-self.Tmin_pnmax-q*self.Topt_pnmax)/((1+q)*self.Topt_pnmax-self.Tmin_pnmax-q*self.Tref)))[(T>self.Topt_pnmax) & (self.pathway=="C4")]
        pnmax_leafN_fac = np.minimum(self.LNC/(self.Nref*1000),1)  #self.LNC in mg/g DM, self.Nref in kg N/kg biomass
        pnmax_adj = self.pnmax*pnmax_T_fac*pnmax_leafN_fac      
        
        #alpha
        alpha_leafN_fac = np.minimum(0.5+0.5*(self.LNC/(self.Nref*1000)),1)
        alpha_T_fac = np.minimum(1-self.lambdaT*(T-self.Topt_alpha),1)
        alpha_T_fac[np.isnan(alpha_T_fac)] = 1 #for C4 species no temp adjustment
        alpha_adj = self.alpha*alpha_leafN_fac*alpha_T_fac
        p_pars = np.array([self.p_model,pnmax_adj,alpha_adj,self.rd,self.angle])
        
        #adapt maintenance respiration based on temperature
        rm_T_fac = (T-self.Tmin_rm)/(self.Tref-self.Tmin_rm)
        rm_leafN_fac = self.LNC/(self.Nref*1000) #same constants as pnmax adjustment
        
        #previous biomass [g/m²]
        biomass_t0 =  self.biomass
        
        #calculate new biomass [g/m²]
        self.height = np.array([growth_input['PFT'+str(i)] for i in range(len(self.species))]) #[cm]
        self.biomass = (1+1/self.SR)*(self.a*self.cover**self.b*self.height**self.c) #[g/m²]
        
        #carbon demand for growth and maintenance [gC/m².day]
        net_growth = (self.biomass-biomass_t0)*constants["C_biomass_ratio"] #[gC/plot.day]
        carbon_demand = net_growth/(1-self.rg)+self.biomass*constants["C_biomass_ratio"]*self.rm*rm_T_fac*rm_leafN_fac #Pn = (1-Rg)*(Pg-Rm*BiomassC) based on Taubert et al., 2012

        #split biomass in plant compartments: [leaves,roots] [g/m²]
        self.mass = np.array([self.biomass]).transpose()*self.architecture
        
        #Calculate potential growth due to available light -> without plasticity
        resolution = self.shoot_layers[0] 
        g_light = np.zeros((len(self.shoot_layers)+1,len(self.species)))
        LA = self.mass[:,0]*self.sla*0.0001 #[m²/m²] 
        LA_h = np.divide(LA, self.height, out=np.zeros(np.shape(LA)), where=self.height!=0) #total LAI per unit of height for each age cohort , return nan for zero biomass pft
        for i,l in enumerate(self.shoot_layers):
            LAI_above_layer = sum((self.k*LA_h*(self.height-l))[self.height>l])
            height_within_layer = np.array([min(resolution,h-(l-resolution)) for h in self.height])
            height_within_layer[height_within_layer<0] = 0 
            LA_within_layer = LA_h*height_within_layer
            LAI_above_layer = LAI_above_layer + self.k*0.5*LA_within_layer #all leaves within layer receive the amount of light that is available in the middle of the layer
            #no competition for light (all leaves receive incoming light at forest floor)
            if light_comp == 'no':LAI_above_layer = LAI_above_layer*0   
            g_light[i,:] = Net_gross_photosynthesis(LA_within_layer,LAI_above_layer,[growth_input['daylength'],growth_input['light']],self.k,p_pars)
        height_within_layer = self.height-self.shoot_layers[-1]
        height_within_layer[height_within_layer<0] = 0 
        LA_within_layer = LA_h*height_within_layer
        LAI_above_layer = self.k*0.5*LA_within_layer #all leaves within layer receive the amount of light that is available in the middle of the layer
        if light_comp == 'no':LAI_above_layer = LAI_above_layer*0
        g_light[-1,:] = Net_gross_photosynthesis(LA_within_layer,LAI_above_layer,[growth_input['daylength'],growth_input['light']],self.k,p_pars)
        g_light = np.sum(g_light,axis=0) #[gC/plot.day]
        daily_photosynthesis = g_light    

        #BELOWGROUND SYMMETRIC COMPETITION
        #---------------------------------
        
        #soil resources
        soil_resources = np.array([[growth_input["soil_W_"+str(i)],growth_input["soil_N_"+str(i)],growth_input["soil_P_"+str(i)]] for i in range(len(self.root_layers))])
        
        #Determine demand for belowground resources        
        W_demand = carbon_demand/self.WUE #water demand expressed as kg or mm water per day
        N_demand = 0.001*carbon_demand*self.LNC/constants["C_biomass_ratio"] #nutrient demand expressed as g N per day
        P_demand = 0.001*carbon_demand*self.LPC/constants["C_biomass_ratio"] #nutrient demand expressed as g P per day
        total_resource_demand = [W_demand,N_demand,P_demand]
        
        #Calculate available belowground resources     
        resolution = self.root_layers[0]
        resource_available = np.zeros((len(self.root_layers),3,len(self.species)))
        rootlength = self.mass[:,1]*self.srl
        rootlength_d = np.divide(rootlength,self.root_depth,out=np.zeros(np.shape(rootlength)),where=self.root_depth!=0)
        for i,l in enumerate(self.root_layers):
            rootdepth_within_layer = np.array([min(resolution,d-(l-resolution)) for d in self.root_depth])
            rootdepth_within_layer[rootdepth_within_layer<0] = 0 
            rootlength_within_layer = rootdepth_within_layer*rootlength_d
            if sum(rootlength_within_layer)==0:
                resource_av_within_layer = np.zeros((3,len(rootlength_within_layer)))
            else: 
                resource_av_within_layer = np.array([r_s*rootlength_within_layer/sum(rootlength_within_layer) for r_s in soil_resources[i,:]])
            resource_available[i] = resource_av_within_layer
        total_resource_available = np.sum(resource_available,axis=0) #sum over layers
        
        #Calculate daily performance based on resource demand and calculated acquisition rates
        total_resource_demand = np.vstack((total_resource_demand,carbon_demand))
        total_resource_available = np.vstack((total_resource_available,g_light)) 
        total_resource_use = total_resource_available 
        #total_resource_use = np.minimum(total_resource_demand,total_resource_available) #for dynamically interacting with the soil
        R = np.divide(total_resource_use,total_resource_demand,out = np.zeros(np.shape(total_resource_use)),where=total_resource_demand!=0)
        R_min = np.min(R,axis=0) #minimum over resource types (N uptake cannot compensate P uptake, or C uptake) 
        limiting_resource = np.argmin(R,axis=0) #water (0), N (1), P (2), light (3)
        daily_performance_value = R[3] #only account for light limitation
        #daily_performance_value = R_min
        #daily_performance_value = carbon_demand*(R_min-1) #calculates difference between expected growth curve and calculated growth curve, if R lower than 1, a high demand will punish performance and vice versa
    
        #when no carbon demand (lower than zero), performance values not insightful anymore and can be set to zero
        daily_performance_value[carbon_demand<0]=0 
        
        return [daily_photosynthesis,daily_performance_value,limiting_resource] #array of performance values, one for each PFT
            
    def Update(self,performance):
        """
        Calculate cover change based on performance
        """
        #set change parameters (to be calibrated)
        A = pars['Dynamics']['A'] 
        B = pars['Dynamics']['B'] 
        
        #calculate cover change
        #coverchange = A*performance + B
        #self.cover = self.cover + coverchange
        self.cover = self.cover
        
    def Age(self):
        
        #move remaining biomass to higher age class      
        for s in self.unique_species: 
            species_cover = self.cover[self.species==s]
            if np.shape(species_cover)!=1: #do noting if there's only one age class
                cover = np.roll(species_cover,1,axis=0)
                cover[-1]+=cover[0];cover[0]=0 #mass accumulates in last group, always no mass in first group, will be introduced by seeding only
                self.cover[self.species==s]=cover
        
class Soil():
    
    def __init__(self, s_layers, s_res_t0, soil_treatment):
        
        """
        Initialize availability of water and nutrients in the different soil layers, s_res_t0 is a 3 element array representing N, P and water availability
        """
        
        #to expand to more layers
        #set initial value
        self.treatment = soil_treatment
        self.layers = Layers(s_layers[0],s_layers[1])
        dim = np.shape(s_res_t0)
        if dim==(3,): #vector with availability of N, P and water
            self.soil_t0 = np.array([np.array(s_res_t0)/len(self.layers)]*len(self.layers))
        elif dim[1]==3: 
            if dim[0]==len(self.layers): #vector per soil layer with availability of N, P and water
                self.soil_t0 = np.array(s_res_t0)
            else: #vector per soil layer with availability of N, P and water, but different amount of soil layers than in model
                s_res_t0 = np.sum(s_res_t0,axis=0)
                self.soil_t0 = np.array([s_res_t0/len(self.layers)]*len(self.layers))
        else:
            raise DynaFORbError("'Initial soil resources'-parameter not in the desired format")    
        

    def Dynamics(self,canopy_dynamics,microclimate_dynamics):
        
        """
        Calculate annual trend of nutrient and water availability
        """
        #initialize annual trend 
        self.dynamics = pd.DataFrame()
        
        self.dynamics['dates'] = microclimate_dynamics['dates']
        l=0
        for i,resource_array in enumerate(self.soil_t0):
            self.dynamics['soil_N_'+str(int(l))] = resource_array[0]
            self.dynamics['soil_P_'+str(int(l))] = resource_array[1]
            self.dynamics['soil_W_'+str(int(l))] = resource_array[2]
            l+=1
            
    def Age(self):
        return
        
class DynaFORbError(Exception):
    
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

#--------------------#
#  HELPER FUNCTIONS  #
#--------------------#

def Layers(number_of_layers,upper_limit):
    resolution = upper_limit/number_of_layers
    return np.arange(resolution,upper_limit+resolution,resolution)

    
#--------------------#
#       MAIN         #
#--------------------#

def sensitivity_run(n=100):
    
    #initialise graph
    import seaborn as sns
    clrs = sns.color_palette("husl", 5)
    
    #define parameters of interest
    runs = [['Alpha'],['Pnmax'],['Rd']]
    
    for j,r in enumerate(runs):
    
        virt_spec = Generate ('Anemone nemorosa',r,n)
        mass = np.zeros((n,365))
        i=0
        
        for s in virt_spec.index:     
            
            #Open plot
            s_plot = Plot(PFTs = virt_spec, h_spec=[s],s_res_t0=[[100,100,100]],veg_layers=[1,50],veg_t0 ="One",c_LAI = 4) #one species, lots of belowground resources, one vegetation layer,one individual                                
            #Run simulation
            s_plot.Simulate([2014,2014],modules=[]) #one year simulation     
            #Yearly mass
            mass[i,:] = s_plot.mass.sum(axis=1,level="Species").values.flatten()
            
            i+=1
        
        #sensitivity plot
        fig, ax = pyplot.subplots()
        mean = mass.mean(axis=0) 
        lb = mass.min(axis=0)
        ub = mass.max(axis=0)
        with sns.axes_style("darkgrid"):
            ax.plot(np.arange(365), mean, label=r, c=clrs[j])
            ax.fill_between(np.arange(365), lb, ub ,alpha=0.3, facecolor=clrs[j])
            ax.legend()
    
    return mass
   
if __name__ == "__main__":
    plot = Plot()
    output = plot.Simulate()    