# DYNAmics of FORest floor Biodiversity

**DynaForB** is a **trait-based** model to predict year-to-year cover changes for understorey species (or plant functional types) as a response to changes in growing conditions, with a focus on light availability. Predicted cover changes are based on yearly carbon gain estimates, calculated on a daily basis, based on a species’ architecture (height, leaf mass fraction, specific leaf area), leaf-level photosynthesis and respiration rates and leaf phenology.

The **architecture** of a species is quantified by a set of plant characteristics, including plant height, leaf mass fraction, specific leaf area and the vertical distribution of leaves. To be able to link these mass-based plant parameters to the model’s state variables, which are cover-based, published allometric relations are used (translating plant cover to above-ground biomass). Based on all these parameters, the model calculates for each species the leaf area in a predefined number of vegetation layers and uses this to calculate competition for light among all co-occurring species.

The **phenology** of a species determines when it’s active and able to take up carbon, but also whether species growing in the same plot are, at a certain point in time, competing for light or not.
