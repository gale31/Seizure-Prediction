%% Getting started
close all; clear;  clc;

dataDir = '/Users/Gale/Documents/kaggle/Seizure-Prediction/data'; % data directory
featureDir =  '/Users/Gale/Documents/kaggle/Seizure-Prediction/features' % feature directory

if ~exist(dataDir,'dir')
    mkdir(dataDir)
elseif ~exist(featureDir,'dir')
    mkdir(featureDir)
end

%% Compute a set of features for the training (and test) data.
%   Features of the training data will be later on used to train a  model. 
%   And the trained model will be used to make predictions based on features of the test data.

%trains = '/train_'
%generate_features(dataDir, featureDir, trains);

% Uncomment the above to generate features, mine are already created.

%% Prepare the data

% X = []
% 
% for i = 1:1302
%     dir = strcat(featureDir, '/train_1_', num2str(i), '.mat');
%     features = load(dir);
%     X = [X; features.average]
% end
% 
% for i = 1:2346
%     dir = strcat(featureDir, '/train_2_', num2str(i), '.mat');
%     features = load(dir);
%     X = [X; features.average]
% end

% Y = []
% 
% ddir = strcat(dataDir, '/train_1');
% dirv = dir([ddir, '/*.mat'])
%     
% for samplefile = dirv'
%     display(samplefile.name);
%     if samplefile.name(end-4:end-4) == '1'
%         Y = [Y, 1];
%     else
%         Y = [Y, 0];
%     end
% end
% 
% ddir = strcat(dataDir, '/train_2');
% dirv = dir([ddir, '/*.mat'])
%     
% for samplefile = dirv'
% %     display(samplefile.name);
%     if samplefile.name(end-4:end-4) == '1'
%         Y = [Y, 1];
%     else
%         Y = [Y, 0];
%     end
% end

trainingData = load('/Users/Gale/Documents/kaggle/Seizure-Prediction/features/trainingData.mat');

%% Train a model based on the data.
%  For our firs try, we are going to use a regression decision tree as it is fast and easy to implement.

rtree = fitrtree(trainingData.X, trainingData.Y)
%view(rtree)
view(rtree,'mode','graph')

%% Predict Y for the test data.

% We'll have to generate features again

%tests = '/test_';
%generate_features(dataDir, featureDir, tests);

% Going through the same steps for X...

% X = []
% 
% for i = 1:216
%     dir = strcat(featureDir, '/test_1_', num2str(i), '.mat');
%     features = load(dir);
%     X = [X; features.average]
% end
% 
% for i = 1:1002
%     dir = strcat(featureDir, '/test_2_', num2str(i), '.mat');
%     features = load(dir);
%     X = [X; features.average]
% end

% X_test = load('/Users/Gale/Documents/kaggle/Seizure-Prediction/features/X_test.mat')
% 
% Y = predict(rtree, X_test.X)
% 
% % Name of test file for every index
% name = load('/Users/Gale/Documents/kaggle/Seizure-Prediction/features/for_test.mat')
% name.for_test = strcat({'new_'}, name.for_test);
% 
% final = [name.for_test num2cell(Y)]
% 
% % writing to csv sucks
% 
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
% TODO: use train_and_test_data_labels_safe