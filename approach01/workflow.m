close all; clear;  clc;

dataDir = '/Users/Gale/Documents/data';
featureDir = '/Users/Gale/Documents/Seizure-Prediction/features';

if ~exist(dataDir,'dir')
    mkdir(dataDir)
elseif ~exist(featureDir,'dir')
    mkdir(featureDir)
end
%% Compute a set of features for the training (and test) data.

% We will compute this set of features for every one of the two (or three, or n) patients

% dirb = 'train_'; patient = '1';
% generate_features(dataDir, featureDir, dirb, patient);
% 
% dirb = 'train_'; patient = '2';
% generate_features(dataDir, featureDir, dirb, patient);

%% Prepare the data

%  X = []
%  
%  average = load(strcat(featureDir, '/average_train_1.mat'));
%  std = load(strcat(featureDir, '/std_train_1.mat'));
%  
% % display(average.average(1,2));
%  
%  for i = 1:720
%      X = [X; average.average(i, 1:16), std.standardDev(i, 1:16)]
%  end
%  
%  savedir = fullfile(featureDir, '/X_1.mat');
%  save(savedir, 'X');
%  
%  Y = [];
%  
%  fileOrder = load(strcat(featureDir, '/fileOrder_train_1.mat'));
%  
%  for i = 1:720
%      
%      split = strsplit(char(fileOrder.fileOrder(i, 1)),'_');
%      
%      if char(split(2)) == '0.mat'
%          Y = [Y; 0];
%      else
%          Y = [Y; 1];
%      end
%      
%  end
%  
%  savedir = fullfile(featureDir, '/Y_1.mat');
%  save(savedir, 'Y')
%   
%  average = load(strcat(featureDir, '/average_train_2.mat'));
%  std = load(strcat(featureDir, '/std_train_2.mat'));
%  
%   for i = 1:1986
%       X = [X; average.average(i, 1:16), std.standardDev(i, 1:16)]
%   end
%   
%  savedir = fullfile(featureDir, '/X_2.mat');
%  save(savedir, 'X');
%   
%   Y = []
%  
%  fileOrder = load(strcat(featureDir, '/fileOrder_train_2.mat'));
%  
%  for i = 1:1986
%      
%      split = strsplit(char(fileOrder.fileOrder(i, 1)),'_');
%      
%      if char(split(2)) == '0.mat'
%          Y = [Y; 0];
%      else
%          Y = [Y; 1];
%      end
%      
%  end
%  
%  savedir = fullfile(featureDir, '/Y_2.mat');
%  save(savedir, 'Y');

% the only thing from above that's useful now is finding Y (we've come a long/complicated way)

%% Train a model based on the data.

% For the first patient.

average_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_train_1.mat');
std_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_train_1.mat');

patient1x = zeros(720, 2, 16);
for i = 1:16
    patient1x(:, :, i) = [average_1.average(:, i) std_1.standardDev(:, i)];
end

Y_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_1.mat');

patient1dtree = cell(16, 1);
for i = 1:16  
    patient1dtree{i} = fitctree(patient1x(:, :, i), Y_1.Y);
end

% view(patient1dtree{3},'mode','graph')

% For the second patient.

average_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_train_2.mat');
std_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_train_2.mat');

patient2x = zeros(1986, 2, 16);
for i = 1:16
    patient2x(:, :, i) = [average_2.average(:, i) std_2.standardDev(:, i)];
end

Y_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_2.mat');

patient2dtree = cell(16, 1);
for i = 1:16  
    patient2dtree{i} = fitctree(patient2x(:, :, i), Y_2.Y);
end

% view(patient1dtree{3},'mode','graph')

%% Predict Y for the test data.

% We'll have to generate features again

% dirb = 'test_'; patient = '1';
% generate_features(dataDir, featureDir, dirb, patient);
%  
% dirb = 'test_'; patient = '2';
% generate_features(dataDir, featureDir, dirb, patient);

average_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_test_1.mat');
std_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_test_1.mat');
  
Y = zeros(216, 16);
for i = 1:16
    X = [average_1.average(:, i) std_1.standardDev(:, i)];
    Y(:, i) = predict(patient1dtree{i}, X);
end

prediction_1 = mean(Y.');

average_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_test_2.mat');
std_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_test_2.mat');
  
Y = zeros(1002, 16);
for i = 1:16
    X = [average_2.average(:, i) std_2.standardDev(:, i)];
    Y(:, i) = predict(patient2dtree{i}, X);
end

prediction_2 = mean(Y.');

% you know what is hard? writing to csv from matlab

% fileID = fopen('/Users/Gale/Documents/kaggle/Seizure-Prediction/submission.csv','wt');
% [rows,cols]=size(final)
% 
% for i = 1:rows
%       fprintf(fileID, '%s,', final{i, 1:1})
%       fprintf(fileID, '%f\n', final{i, 2})
% end
% 
% fclose(fileID);

%%
% TODO: remove data dropouts later