function generate_features(dataDir, featureDir, trainortest)
% This function generates a set of features which are later used to train a model.
% Here the features are simply the average for the whole signal in each channel.

% Read data, compute and save features
% Warning: I have downloaded only train_1 and train_2, so these are the
% only data files I'm going to extract features from.

for_test = [];
for i = 1:2
    trainingdir = fullfile(dataDir, strcat(trainortest, num2str(i)));
    dirv = dir([trainingdir, '/*.mat'])
    
    j = 0;
    for samplefile = dirv'
        
        split = strsplit(samplefile.name,'_')
        for_test = [for_test; strcat(split(2), '_', split(3))];
      
        j = j + 1;
         
        sample = load(fullfile(trainingdir, samplefile.name));
        
        average = zeros(1, 16);
        
        for channel = 1:16
            chanarr = sample.dataStruct.data(:, channel);
            average(1, channel) = mean(chanarr);
        end
        
        savedir = fullfile(featureDir, strcat(trainortest, num2str(i), '_', num2str(j), '.mat'));
       
        display(savedir);
        save(savedir, 'average');
    end
end

savedir = fullfile(featureDir, '/for_test.mat');
save(savedir, 'for_test');

end