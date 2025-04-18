# Next-Generation-of-Offshore-Wind-Turbines
This repository contains the complete Python-based modelling and optimisation framework developed for the redesign of the Sheringham Shoal offshore wind farm using next-generation 25 MW turbines. It forms part of a comprehensive engineering design project targeting improved energy yield and reduced lifetime levelised cost of energy (LCOE).

The framework integrates multiple key components:
	•	Wind Resource Modelling: A logarithmic-altitude-based wind characterisation using data from the Global and European Wind Atlases, with fitted Weibull distributions for any hub height.
	•	Wake Effect Simulation: A scalable implementation of Jensen’s single and multiple wake models with directional wind variation, enabling the calculation of turbine-specific Weibull parameters for all incoming wind angles.
	•	Energy Yield Estimation: Directionally weighted annual energy production (AEP) is computed using Blade Element Momentum (BEM) spreadsheets linked via xlwings.
	•	Layout Optimisation: Two separate optimisation schemes:
	•	Irregular layout: Genetic algorithm (GA) with tailored crossover and mutation functions, producing free-form layouts.
	•	Regular layout: Exhaustive grid search over a defined 3D parameter space (row spacing, column spacing, orientation) for structured parallelogram arrangements.
	•	LCOE Estimation: Custom function implementing the lifetime LCOE equation using site-specific CAPEX and OPEX assumptions, allowing objective comparison between different layouts.

The code is highly modular, easily adaptable to any offshore site by adjusting the boundary geometry, wind input data, and turbine design inputs.
