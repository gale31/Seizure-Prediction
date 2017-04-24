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
 
% dirb = 'train_'; patient = '2';
% generate_features(dataDir, featureDir, dirb, patient);
 
% dirb = 'train_'; patient = '3';
% generate_features(dataDir, featureDir, dirb, patient);

%% Prepare Y
  
%  Y = [];
%  
%  fileOrder = load(strcat(featureDir, '/fileOrder_train_1.mat'));
%  
%  for i = 1:720
%      
%      split = strsplit(char(fileOrder.file_order(i, 1)),'_');
%      
%      if char(split(3)) == '0.mat'
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
%  Y = [];
%  
%  fileOrder = load(strcat(featureDir, '/fileOrder_train_2.mat'));
%  
%  for i = 1:1986
%      
%      split = strsplit(char(fileOrder.file_order(i, 1)),'_');
%      
%      if char(split(3)) == '0.mat'
%          Y = [Y; 0];
%      else
%          Y = [Y; 1];
%      end
%      
%  end
%  
%  savedir = fullfile(featureDir, '/Y_2.mat');
%  save(savedir, 'Y');
% 
% Y = [];
%  
% fileOrder = load(strcat(featureDir, '/fileOrder_train_3.mat'));
%  
% for i = 1:2058
%     split = strsplit(char(fileOrder.file_order(i, 1)),'_');
%      
%     if char(split(3)) == '0.mat'
%         Y = [Y; 0];
%     else
%         Y = [Y; 1];
%     end 
% end
%  
% savedir = fullfile(featureDir, '/Y_3.mat');
% save(savedir, 'Y');

%% Train a model based on the data.
 
% For the first patient.

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_train_1.mat');
features = features_file.features;
Y_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_1.mat');
Y = Y_1.Y;

patient1dtree = cell(16, 1);
for i = 1:16  
    patient1dtree{i} = fitctree([features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)], Y);
end

% For the second patient.

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_train_2.mat');
features = features_file.features;
Y_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_2.mat');
Y = Y_2.Y;

patient2dtree = cell(16, 1);
for i = 1:16  
    patient2dtree{i} = fitctree([features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)], Y);
end

% And, finally, for the third patient.

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_train_3.mat');
features = features_file.features;
Y_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_3.mat');
Y = Y_3.Y;

patient3dtree = cell(16, 1);
for i = 1:16  
    patient3dtree{i} = fitctree([features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)], Y);
end

%view(patient3dtree{3},'mode','graph');
%% Predict for the test data.
 
% We'll have to generate features again

% dirb = 'test_'; patient = '1';
% generate_features(dataDir, featureDir, dirb, patient);
% 
% dirb = 'test_'; patient = '2';
% generate_features(dataDir, featureDir, dirb, patient);
% 
% dirb = 'test_'; patient = '3';
% generate_features(dataDir, featureDir, dirb, patient);

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_test_1.mat');
features = features_file.features;
  
Y = zeros(216, 16);
for i = 1:16
    Y(:, i) = predict(patient1dtree{i}, [features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)]);
end

prediction_1 = zeros(216);
for i = 1:216
    if sum(Y(i, :)) > 7
        prediction_1(i) = 0;
    else 
        prediction_1(i) = 1;
    end
end

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_test_2.mat');
features = features_file.features;
  
Y = zeros(1002, 16);
for i = 1:16
    Y(:, i) = predict(patient1dtree{i}, [features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)]);
end

prediction_2 = zeros(1002);
for i = 1:1002
    if sum(Y(i, :)) > 7
        prediction_2(i) = 0;
    else 
        prediction_2(i) = 1;
    end
end

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_test_3.mat');
features = features_file.features;
  
Y = zeros(690, 16);
for i = 1:16
    Y(:, i) = predict(patient1dtree{i}, [features.avmean(:, i), features.std(:, i), features.skewness(:, i), features.kurtosis(:, i), features.activity(:, i), features.mobility(:, i), features.complexity(:, i), features.shentropy(:, i), features.spedge(:, i), features.shentropyDyd(:, i)]);
end

prediction_3 = zeros(690);
for i = 1:690
    if sum(Y(i, :)) > 7
        prediction_3(i) = 0;
    else 
        prediction_3(i) = 1;
    end
end
 
fileID = fopen('/Users/Gale/Documents/Seizure-Prediction/approach03/submission.csv','wt');

fileOrder_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_1.mat');
fileOrder_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_2.mat');
fileOrder_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_3.mat');
 
display('Writing to csv...');

for i = 1:1908
    if i <= 216
        fprintf(fileID, '%s,%f\n', char(fileOrder_1.file_order(i)), double(prediction_1(i)));
    elseif i <= 1218
        fprintf(fileID, '%s,%f\n', char(fileOrder_2.file_order(i-216)), double(prediction_2(i-216)));
    else
        fprintf(fileID, '%s,%f\n', char(fileOrder_3.file_order(i-1218)), double(prediction_3(i-1218)));
    end
end

display('Mission accomplished.');

fclose(fileID);