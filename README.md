# fMRI_connectivity_accurately_distinguishes_cases
Code to enable reproduction of headline results in 'Functional MRI connectivity accurately distinguishes cases with psychotic disorders from healthy controls, based on cortical features associated with brain network development'

# Requirements
This code runs under Matlab and also requires access to the GPML library. This can be obtained from http://www.gaussianprocess.org/gpml/code/matlab/doc/.

# Usage
As described in the paper, the cross validation strategy we used was a 200-fold randomized cross validation. The classify.m file reads in functional
connectivity data for all three datasets used in the paper, as well as their associated metadata (case/control labels, age, sex, and estimated level of motion during the scan)
classify.m reads in the selected data and metadata, applies corrections for age, sex and motion (assessed with mean framewise displacement) using the method described in the paper 
(see also https://www.sciencedirect.com/science/article/pii/S1053811914003371), performs a single iteration of the cross validation, and writes files
containing probabilistic predictions for case/control status to file. Files are also output to store predictions of age, sex, and motion level,
to enable some checks on how well the correction methods are working. The output format is as follows:

Subject ID | case/control label | age | sex | motion | prediction

where prediction is a probabilistic prediction of case or male status for prediction of case/control or prediction of sex files respectively,
or point predictions of age or motion level for age and motion level prediction files. To enable assessment of which connections or regions
are important in the classification, we also output files containing the classification/regression weights for each fold and each prediction tas.

As the correction method is quite slow, we recommend using a cluster if available to run a large number of cross validation folds in parallel,
using the job id to index the output files and seed the random number generator so each train/test split is different. Alternatively classify.m
can easily be modified to perform a number of cross validations iteratively on a single machine. In this case output files could either be per fold
or a single file containing all predictions.
