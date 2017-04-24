function generate_features(dataDir, featureDir, trainOrTest, patientNum)

fileOrder = [];

% fileOrder because "dir([trainDir, '/*.mat'])" doesn't give them in the
% wanted order

trainDir = fullfile(dataDir, strcat('/', trainOrTest, patientNum));
matFiles = dir([trainDir, '/*.mat']);
numMatFiles = length(matFiles(not([matFiles.isdir])));

average = zeros(numMatFiles, 16);
standardDev = zeros(numMatFiles, 16);
    
i = 0;
for sampleFile = matFiles'
    
    display('Going through...');
        
    fileOrder = [fileOrder; cellstr(sampleFile.name)];
    
    i = i + 1;
         
    sample = load(fullfile(trainDir, sampleFile.name));
        
    for channel = 1:16
        channelArray = sample.dataStruct.data(:, channel);
        average(i, channel) = mean(channelArray);
        standardDev(i, channel) = std2(channelArray);
    end
    
end

display('Saving...');

savedir = fullfile(featureDir, strcat('/fileOrder_', trainOrTest, patientNum,'.mat'));
save(savedir, 'fileOrder');
% fileOrder-a za generirane na suotvetnite features

savedir = fullfile(featureDir, strcat('/average_', trainOrTest, patientNum,'.mat'));
save(savedir, 'average');

savedir = fullfile(featureDir, strcat('/std_', trainOrTest, patientNum,'.mat'));
save(savedir, 'standardDev');

display('Mission accomplished.');

end