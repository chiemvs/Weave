# Weave
Empiricial tools for long-range forecasts of air temperature.
Searches potential sources of predictability in other climate variables (like sea surface temperature and snow-cover). 
Builds a predictive model to integrate information from multiple time scales.
Applies two explainable AI tools to distinguish the real sources of predictability from the potential ones.
Quantifies usefulness of the empirical forecasts with multiple metrics against simple benchmarks.

### Application
Created for a study into the predictability of high European summer temperature.
Specific focus on sub-seasonal lead times (2+ weeks).
Fully based on the ERA5 and ERA5-Land reanalysis.

### Main features
* Downloader for ERA5 data from the Copernicus Climate Data Store (CDS).
* Hierarchical clustering to find region in which temperature might be predictable.
* Three-step dimension reduction for each climate variable: association + clustering + extraction.
* Multi-processing for grid-cell operations on large files.  
* ML based on Random Forest: number of features > number of independent samples.
* Separate learning of climate change trend and (predictable) deviations.
* Explainable AI with Permutation Importance and TreeSHAP.

### Package structure
* *Weave/* Source code for classes and computations
	* *association.py* Apply association measures like rank-correlation (step 1)
	* *clustering.py* General Cluster class for temperature-region finding and for (step 2) 
	* *dimreduction.py* Extract 1D timeseries from 3D data (step 3)
	* *downloaders.py* ERA5 downloader
	* *inputoutput.py* read-write functionality for storing to disk (netCDF)
	* *inspection.py* Explainable AI
	* *models.py* Machine learning models
	* *processing.py* Pre-processing, remove seasonality, create anomalies, time-aggregation
	* *utils.py* Additional functions plus some verification
* *hpc/* Scripts that start large computations and contain most parameters
	* ...
* *scripts/* Scripts that start small computations
	* *era5_downloads.py* manage ERA5 downloads
	* ...
* *notebooks/* Visualization and exploration of results 
	* *Classifier_skills.ipynb* Verification of ML model 
	* *Dimred.ipynb* Verification of ML model 
	* *Geographical_illustration.ipynb* geographic visualization of explainable AI results
	* *inspection_showcase.ipynb* inspection of ML model with explainable AI
	* *investigate.ipynb* explorative version of inspection_showcase
	* *Parcor_cluster_params.ipynb* iterative exploration of HDBSCAN parameters
	* ...
* *tests/* pytest directory (out-dated)
	* ...

