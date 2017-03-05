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

model_1patient = fitrensemble([features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)], Y);

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_train_2.mat');
features = features_file.features;
Y_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_2.mat');
Y = Y_2.Y;

model_2patient = fitrensemble([features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)], Y);

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_train_3.mat');
features = features_file.features;
Y_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_3.mat');
Y = Y_3.Y;

model_3patient = fitrensemble([features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)], Y);

%view(patient1dtree,'mode','graph');

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

prediction_1 = predict(model_1patient, [features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)]);
   
features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_test_2.mat');
features = features_file.features;

prediction_2 = predict(model_2patient, [features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)]);

features_file = load('/Users/Gale/Documents/Seizure-Prediction/features/features_test_3.mat');
features = features_file.features;

prediction_3 = predict(model_3patient, [features.avmean(:, :), features.std(:, :), features.skewness(:, :), features.kurtosis(:, :), features.activity(:, :), features.mobility(:, :), features.complexity(:, :), features.shentropy(:, :), features.spedge(:, :), features.shentropyDyd(:, :)]);
 
fileID = fopen('/Users/Gale/Documents/Seizure-Prediction/approach05/submission.csv','wt');

fileOrder_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_1.mat');
fileOrder_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_2.mat');
fileOrder_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_3.mat');
 
display('Writing to csv...');

for i = 1:1908
    if i <= 216
        if prediction_1(i) < 0
            fprintf(fileID, '%s,%f\n', char(fileOrder_1.file_order(i)), double(0));
        else
            fprintf(fileID, '%s,%f\n', char(fileOrder_1.file_order(i)), double(prediction_1(i)));
        end
    elseif i <= 1218
        if prediction_2(i-216) < 0
            fprintf(fileID, '%s,%f\n', char(fileOrder_2.file_order(i-216)), double(0));
        else
            fprintf(fileID, '%s,%f\n', char(fileOrder_2.file_order(i-216)), double(prediction_2(i-216)));
        end
    else
        if prediction_3(i-1218) < 0
            fprintf(fileID, '%s,%f\n', char(fileOrder_3.file_order(i-1218)), double(0));
        else
            fprintf(fileID, '%s,%f\n', char(fileOrder_3.file_order(i-1218)), double(prediction_3(i-1218)));
        end
    end
end

display('Mission accomplished.');

fclose(fileID);