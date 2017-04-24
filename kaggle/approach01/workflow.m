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

dirb = 'train_'; patient = '1';
generate_features(dataDir, featureDir, dirb, patient);
 
dirb = 'train_'; patient = '2';
generate_features(dataDir, featureDir, dirb, patient);

dirb = 'train_'; patient = '3';
generate_features(dataDir, featureDir, dirb, patient);

%% Prepare Y
  
 Y = [];
 
 fileOrder = load(strcat(featureDir, '/fileOrder_train_1.mat'));
 
 for i = 1:720
     
     split = strsplit(char(fileOrder.fileOrder(i, 1)),'_');
     
     if char(split(2)) == '0.mat'
         Y = [Y; 0];
     else
         Y = [Y; 1];
     end
     
 end
 
 savedir = fullfile(featureDir, '/Y_1.mat');
 save(savedir, 'Y')
  
 Y = [];
 
 fileOrder = load(strcat(featureDir, '/fileOrder_train_2.mat'));
 
 for i = 1:1986
     
     split = strsplit(char(fileOrder.fileOrder(i, 1)),'_');
     
     if char(split(2)) == '0.mat'
         Y = [Y; 0];
     else
         Y = [Y; 1];
     end
     
 end
 
 savedir = fullfile(featureDir, '/Y_2.mat');
 save(savedir, 'Y');

Y = [];
 
fileOrder = load(strcat(featureDir, '/fileOrder_train_3.mat'));
 
for i = 1:2058
    split = strsplit(char(fileOrder.fileOrder(i, 1)),'_');
     
    if char(split(2)) == '0.mat'
        Y = [Y; 0];
    else
        Y = [Y; 1];
    end 
end
 
savedir = fullfile(featureDir, '/Y_3.mat');
save(savedir, 'Y');

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

% view(patient1dtree{3},'mode','graph');

% And, finally, for the third patient.

average_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_train_3.mat');
std_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_train_3.mat');

patient3x = zeros(2058, 2, 16);
for i = 1:16
    patient3x(:, :, i) = [average_3.average(:, i) std_3.standardDev(:, i)];
end

Y_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/Y_3.mat');

patient3dtree = cell(16, 1);
for i = 1:16  
    patient3dtree{i} = fitctree(patient3x(:, :, i), Y_3.Y);
end

%view(patient3dtree{3},'mode','graph');

%% Predict for the test data.

% We'll have to generate features again

dirb = 'test_'; patient = '1';
generate_features(dataDir, featureDir, dirb, patient);

dirb = 'test_'; patient = '2';
generate_features(dataDir, featureDir, dirb, patient);

dirb = 'test_'; patient = '3';
generate_features(dataDir, featureDir, dirb, patient);

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

average_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/average_test_3.mat');
std_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/std_test_3.mat');
  
Y = zeros(690, 16);
for i = 1:16
    X = [average_3.average(:, i) std_3.standardDev(:, i)];
    Y(:, i) = predict(patient3dtree{i}, X);
end

prediction_3 = mean(Y.');

fileID = fopen('/Users/Gale/Documents/Seizure-Prediction/approach01/submission.csv','wt');

fileOrder_1 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_1.mat');
fileOrder_2 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_2.mat');
fileOrder_3 = load('/Users/Gale/Documents/Seizure-Prediction/features/fileOrder_test_3.mat');

display('Writing to csv...');

for i = 1:1908
    if i <= 216
        fprintf(fileID, '%s,%f\n', char(fileOrder_1.fileOrder(i)), double(prediction_1(i)));
    elseif i <= 1218
        fprintf(fileID, '%s,%f\n', char(fileOrder_2.fileOrder(i-216)), double(prediction_2(i-216)));
    else
        fprintf(fileID, '%s,%f\n', char(fileOrder_3.fileOrder(i-1218)), double(prediction_3(i-1218)));
    end
end

display('Mission accomplished.');

fclose(fileID);