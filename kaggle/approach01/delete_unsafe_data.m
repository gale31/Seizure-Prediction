fileLabels = readtable('/Users/Gale/Documents/Seizure-Prediction/train_and_test_data_labels_safe.csv');

for k=1:height(fileLabels(:, 1))
     if fileLabels(k, 3).safe == 0
         sampleFile = char(fileLabels(k, 1).image);
         split = strsplit(sampleFile,'_');
         dir = char(strcat('/Users/Gale/Documents/Seizure-Prediction/data/train_', split(1), '/',sampleFile));
         delete(dir);
     end
end 


