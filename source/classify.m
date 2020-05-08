% set directories
data_dir = '../data/';
metadata_dir = '../metadata/';
results_dir = '../results/results/';
weights_dir = '../results/weights/';

% set connectivity data and demographics
data_file = 'Dublin_connectivity_data.mat';
demographics_file = 'Dublin_demographics.csv';

% job index
% in a cluster would get this from array job id eg:
%job_ind = str2num(getenv('SGE_TASK_ID'));
job_ind = 1;

% which group do we want to train remval of potential confounds (age, sex , motion) on?
train_group = 'con';

% do we regress motion alongside age and sex?
regress_motion = true;

% extract data file basename so we know what to call output files
data_basename = strsplit(data_file, '.');
data_basename = data_basename{1};

% read in some data and get data characteristics
connectivity_data = load([data_dir, data_file]);
connectivity_data = connectivity_data.connectivity_data;
[n_subjects, n_connections] = size(connectivity_data);
n_regions = sqrt(n_connections);
tril_inds = find(~tril(ones(n_regions)));
n_features = length(tril_inds);

%  take matrix logs to map connectivity matrices to tangent space at the identity
% then take lower triangle only as matrices are symmetric
logm_connectivity_data = zeros(n_subjects, length(tril_inds));

% loop through the subjects
for i = 1:n_subjects
    
    connectivity_matrix = reshape(connectivity_data(i, :), n_regions, n_regions);
    logm_connectivity_matrix = logm(connectivity_matrix);
    logm_connectivity_vector = reshape(logm_connectivity_matrix, 1, n_connections); 
    logm_connectivity_data(i, :) = logm_connectivity_vector(tril_inds);
    
end

% read in demographics
demographics = readtable([metadata_dir demographics_file]);

% generate split
% use job id to seed random number generator
% so splits on cluster are not identical
rng(job_ind);
perm = randperm(n_subjects);
test_fraction = 0.1;
test_size = int16(n_subjects * test_fraction);
train_size = n_subjects - test_size;
train_inds = perm(1:train_size);
test_inds = perm(train_size + 1:end);

% split labels, data and demographics into train and test
labels = demographics.labels;
motion = demographics.motion;
sex = demographics.sex;
age = demographics.age;
subjects = demographics.SubjectID;
if regress_motion 
    
    demographics = [age sex motion];

else
    
    demographics = [age sex];
    
end
test_subjects = subjects(test_inds);
train_labels = labels(train_inds);
test_labels = labels(test_inds);
train_data = logm_connectivity_data(train_inds, :);
test_data = logm_connectivity_data(test_inds, :);
train_age = age(train_inds);
test_age = age(test_inds);
train_sex = sex(train_inds);
test_sex = sex(test_inds);
train_motion = motion(train_inds);
test_motion = motion(test_inds);
train_demographics = demographics(train_inds, :);
test_demographics = demographics(test_inds, :);

% normalise demographic predictors
test_demographics = bsxfun(@minus, test_demographics, min(train_demographics));
train_demographics = bsxfun(@minus, train_demographics, min(train_demographics));
test_demographics = bsxfun(@rdivide, test_demographics, max(train_demographics));
train_demographics = bsxfun(@rdivide, train_demographics, max(train_demographics));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFOUND REMOVAL %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set a label for subjects to train regression
if strcmp(train_group, 'con')

    regression_label = -1;
    
elseif strcmp(train_group, 'pat')
    
    regression_label = 1;
    
else
    
    regression_label = 0;
    
end



% fit a GP model to each connectivity feature in turn
for i = 1:n_features
    
        if mod(i, 10) == 0
            
            disp(i)
            
        end
        
    
        % targets are the i'th column of features of training data
        targets = train_data(:, i);

        % initialise GP
        % use covariance function from 'Correction of inter-scanner and
        % within-subject variance in structural MRI based automated diagnosing'
        % by Kostro et al, Neuroimage, 2014
        % initial hyperparameter vals:
        % log of median pairwise (training) distance for covSE
        % 0 for covConst
        % 0 (= log(1) ) for all covariance scaling weights
        dists = pdist(train_demographics);
        meanfunc = @meanConst; hyp.mean = 0;
        likfunc = @likGauss; hyp.lik = log(std(targets));
        cov_lin={@covLIN};
        cov_SE={@covSEisoU}; hyp_cov_SE = log(median(dists(:)));
        cov_const={@covConst}; hyp_cov_const = 0;
        cov_eye={@covEye};
        cov_lin_scaled={@covScale,{cov_lin{:}}}; hyp_cov_lin_scaled = 0;
        cov_SE_scaled={@covScale,{cov_SE{:}}}; hyp_cov_SE_scaled = [0 hyp_cov_SE];
        cov_eye_scaled={@covScale,{cov_eye{:}}}; hyp_cov_eye_scaled = 0;
        cov_sum_scaled={@covSum,{cov_lin_scaled,cov_SE_scaled,cov_const, cov_eye_scaled}}; hyp_cov_sum_scaled = [hyp_cov_lin_scaled hyp_cov_SE_scaled hyp_cov_const hyp_cov_eye_scaled];
        hyp.cov = hyp_cov_sum_scaled;
        covfunc = cov_sum_scaled;

        if regression_label == 0
        
           % optimise the hyperparameters
            hyp_opt = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, train_demographics, targets);

            % make predictions for train data
            [mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_demographics, targets, train_demographics);

            % overwrite original features with residuals
            train_data(:, i) =  train_data(:, i) - mu;

            % make predictions for test data
            [mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_demographics, targets, test_demographics);

            % overwrite original features with residuals
            test_data(:, i) =  test_data(:, i) - mu;
        
      else
    
            % optimise the hyperparameters
            hyp_opt = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, train_demographics(train_labels==regression_label, :), targets(train_labels==regression_label));

            % make predictions for train data
            [mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_demographics(train_labels==regression_label, :), targets(train_labels==regression_label), train_demographics);

            % overwrite original features with residuals
            train_data(:, i) =  train_data(:, i) - mu;

            % make predictions for test data
            [mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_demographics(train_labels==regression_label, :), targets(train_labels==regression_label), test_demographics);

            % overwrite original features with residuals
            test_data(:, i) =  test_data(:, i) - mu;
    
       end

       % clear hyp so we can use GP again
       clear hyp
       clear hyp_opt
             
end

% initialise data structures to hold weights and results for
% classification
% store results in following columns:
% class | age | sex | motion | prediction
class_preds = zeros(test_size, 5);
sex_preds = zeros(test_size, 5);
age_preds = zeros(test_size, 5);
motion_preds = zeros(test_size, 5);
class_training_weights = zeros(1, n_features);
age_training_weights = zeros(1, n_features);
sex_training_weights = zeros(1, n_features);
motion_training_weights = zeros(1, n_features);

% fill in fixed portions of results storage - true label, age and sex
class_preds(:, 1) = test_labels;
class_preds(:, 2) = test_age;
class_preds(:, 3) = test_sex;
class_preds(:, 4) = test_motion;
age_preds(:, 1) = test_labels;
age_preds(:, 2) = test_age;
age_preds(:, 3) = test_sex;
age_preds(:, 4) = test_motion;
sex_preds(:, 1) = test_labels;
sex_preds(:, 2) = test_age;
sex_preds(:, 3) = test_sex;
sex_preds(:, 4) = test_motion;
motion_preds(:, 1) = test_labels;
motion_preds(:, 2) = test_age;
motion_preds(:, 3) = test_sex;
motion_preds(:, 4) = test_motion;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% CLASS PREDICTION %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% initialise GP
meanfunc = @meanConst; hyp.mean = 0;
likfunc = @likErf;
covfunc = @covLINone; hyp.cov = 0;

% optimise the hyperparameters
hyp_opt = minimize(hyp, @gp, -200, @infEP, meanfunc, covfunc, likfunc, train_data, train_labels);
    
% make and store predictions
[a b c d lp] = gp(hyp_opt, @infEP, meanfunc, covfunc, likfunc, train_data, train_labels, test_data, ones(test_size, 1));
class_preds(:, 5) = exp(lp);
  
% calculate 'weights' in the manner shown in Quantitative prediction of
% subjective pain intensity from whole-brain fMRI data using Gaussian 
% processes (regression) & Quantifying the Information Content of Brain 
% Voxels Using Target Information, Gaussian Processes and Recursive 
% Feature Elimination (classification), both by Marquand et al
    
% calculate the kernel
K = covfunc(hyp.cov, train_data, train_data);
    
% calculate MAP weights
l2 =  exp(2 * hyp.cov);
weights = (1/l2) * train_data' * (inv(K) * train_labels);
class_training_weights(1, :) = weights;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  AGE PREDICTION %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
meanfunc = @meanConst; hyp.mean = 0;
likfunc = @likGauss; hyp.lik = 0.05;
covfunc = @covLINone; hyp.cov = 0;

% optimise the hyperparameters
hyp_opt = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, train_data, train_age);
    
% make and store predictions
[mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_data, train_age, test_data);
age_preds(:, 5) = mu;
    
% calculate the kernel
K = covfunc(hyp.cov, train_data, train_data);
    
% calculate C
noise_var = exp(2 * hyp.lik);
C = K + noise_var * eye(size(K));
l2 =  exp(2 * hyp.cov);
    
% calculate MAP weights
weights = (1/l2) * train_data' * (inv(C) * train_age);
age_training_weights(1, :) = weights;
   
% clear hyp so we can we have correct number of hyperparameters on next
% iteration
clear hyp
clear hyp_opt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% SEX PREDICTION %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% initialise GP
meanfunc = @meanConst; hyp.mean = 0;
likfunc = @likErf;
covfunc = @covLINone; hyp.cov = 0;
     
% optimise the hyperparameters
hyp_opt = minimize(hyp, @gp, -200, @infEP, meanfunc, covfunc, likfunc, train_data, train_sex);
    
% make and store predictions
[a b c d lp] = gp(hyp_opt, @infEP, meanfunc, covfunc, likfunc, train_data, train_sex, test_data, ones(test_size, 1));
sex_preds(:, 5) = exp(lp);
   
% calculate 'weights' in the manner shown in Quantitative prediction of
% subjective pain intensity from whole-brain fMRI data using Gaussian 
% processes (regression) & Quantifying the Information Content of Brain 
% Voxels Using Target Information, Gaussian Processes and Recursive 
% Feature Elimination (classification), both by Marquand et al
    
% calculate the kernel
K = covfunc(hyp.cov, train_data, train_data);
    
% calculate MAP weights
l2 =  exp(2 * hyp.cov);
weights = (1/l2) * train_data' * (inv(K) * train_sex);
sex_training_weights(1, :) = weights;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% MOTION PREDICTION %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
meanfunc = @meanConst; hyp.mean = 0;
likfunc = @likGauss; hyp.lik = 0.05;
covfunc = @covLINone; hyp.cov = 0;

% optimise the hyperparameters
hyp_opt = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, train_data, train_motion);
    
% make and store predictions
[mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, train_data, train_motion, test_data);
motion_preds(:, 5) = mu;
    
% calculate the kernel
K = covfunc(hyp.cov, train_data, train_data);
    
% calculate C
noise_var = exp(2 * hyp.lik);
C = K + noise_var * eye(size(K));
l2 =  exp(2 * hyp.cov);
    
% calculate MAP weights
weights = (1/l2) * train_data' * (inv(C) * train_motion);
motion_training_weights(1, :) = weights;
   
% clear hyp so we can we have correct number of hyperparameters on next
% iteration
clear hyp
clear hyp_opt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% SAVE RESULTS & WEIGHTS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%

if regress_motion
    
    confound_regression_string = '_asfd_corrected';
    
else
    
    confound_regression_string = '_as_corrected';
    
end

T = array2table(class_preds);
T(:, 6) = cell2table(test_subjects);
T.Properties.VariableNames = {'class' 'age' 'sex' 'motion' 'predicted_p_of_schiz' 'SubjectID'} ;
writetable(T, [results_dir, data_basename, '_', train_group, confound_regression_string, '_logm_class_results_', num2str(job_ind), '.csv']);
T = array2table(sex_preds);
T(:, 6) = cell2table(test_subjects);
T.Properties.VariableNames = {'class' 'age' 'sex' 'motion' 'predicted_p_of_male' 'SubjectID'};
writetable(T, [results_dir, data_basename, '_', train_group, confound_regression_string, '_logm_sex_results_', num2str(job_ind), '.csv']);
T = array2table(age_preds);
T(:, 6) = cell2table(test_subjects);
T.Properties.VariableNames = {'class' 'age' 'sex' 'motion' 'predicted_age' 'SubjectID'};
writetable(T, [results_dir, data_basename, '_', train_group, confound_regression_string, '_logm_age_results_', num2str(job_ind), '.csv']);
T = array2table(motion_preds);
T(:, 6) = cell2table(test_subjects);
T.Properties.VariableNames = {'class' 'age' 'sex' 'motion' 'predicted_motion' 'SubjectID'};
writetable(T, [results_dir, data_basename, '_', train_group, confound_regression_string, '_logm_motion_results_', num2str(job_ind), '.csv']);

csvwrite([weights_dir, data_basename, '_', train_group, confound_regression_string, '_logm_class_weights_', num2str(job_ind), '.csv'], class_training_weights);
csvwrite([weights_dir, data_basename, '_', train_group, confound_regression_string, '_logm_sex_weights_', num2str(job_ind), '.csv'], sex_training_weights);
csvwrite([weights_dir, data_basename, '_', train_group, confound_regression_string, '_logm_age_weights_', num2str(job_ind), '.csv'], age_training_weights);
csvwrite([weights_dir, data_basename, '_', train_group, confound_regression_string, '_logm_motion_weights_', num2str(job_ind), '.csv'], motion_training_weights);

