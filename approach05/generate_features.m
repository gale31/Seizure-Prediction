function generate_features(data_dir, feature_dir, train_or_test, patient_num)

file_order = [];     % fileOrder because "dir([trainDir, '/*.mat'])" doesn't give them in the wanted order

train_dir = fullfile(data_dir, strcat('/', train_or_test, patient_num));
mat_files = dir([train_dir, '/*.mat']);
num_mat_files = length(mat_files(not([mat_files.isdir])));

features = struct('avmean', zeros(num_mat_files, 16), 'std', zeros(num_mat_files, 16), 'skewness', zeros(num_mat_files, 16), 'kurtosis', zeros(num_mat_files, 16), 'activity', zeros(num_mat_files, 16), 'mobility', zeros(num_mat_files, 16), 'complexity', zeros(num_mat_files, 16), 'shentropy', zeros(num_mat_files, 16), 'spedge', zeros(num_mat_files, 16), 'shentropyDyd', zeros(num_mat_files, 16));

i = 0;
for sample_file = mat_files'
    
    display('Going through...');
        
    file_order = [file_order; cellstr(sample_file.name)];
    
    i = i + 1;
         
    sample = load(fullfile(train_dir, sample_file.name));
    data = sample.dataStruct.data;
    [nt,nc] = size(data);
    sampling_rate = sample.dataStruct.iEEGsamplingRate;
    samples_segment = sample.dataStruct.nSamplesSegment;
    
    avmean = mean(data);   % calculate mean for every channel
    standard_dev = std2(data);   % calculate standard deviation for every channel
    skew = skewness(data);   % calculate skewness for every channel
    kurt = kurtosis(data);   % calculate kurtosis for every channel
    activity = var(data);   % activity
    mobility = std(diff(data))./std(data);  % mobility
    complexity = std(diff(diff(data)))./std(diff(data))./mobility; % complexity
    
    data = abs(fft(data));   % take Fast Fourier Transform of each channel
    data = bsxfun(@rdivide, data, sum(data));  % normalize each channel (by the sum)
    hz = [0.1 4 8 14 32 70 180];   % frequency levels in Hz
    hzseg = round(samples_segment / sampling_rate * hz) + 1;  % segments corresponding to frequency bands
    
    dspect = zeros(length(hz) - 1, nc);
    for n = 1:length(hz) - 1
        dspect(n, :) = 2 * sum(data(hzseg(n):hzseg(n+1), :));
    end
    
    shentropy = -sum((dspect) .* log(dspect)); % find the Shannon's entropy
    
    sfreq = sampling_rate;
    tfreq = 40;
    ppow = 0.5;

    topfreq = round(samples_segment/sfreq * tfreq);
    A = cumsum(data(1:topfreq, :));
    B = bsxfun(@minus, A, max(A) * ppow);
    [~,spedge] = min(abs(B));
    spedge = (spedge-1)/(topfreq-1)*tfreq;  % find the spectral edge frequency
    
    ldat = floor(sampling_rate/2);
    no_levels = floor(log2(ldat));  % find the number of dyadic levels
    
    dspect = zeros(no_levels,nc);
    for n = no_levels:-1:1  % find the power spectrum at each dyadic level
         dspect(n,:) = 2*sum(data(floor(ldat/2)+1:ldat,:));
         ldat = floor(ldat/2);
    end
  
    shentropyDyd = -sum(dspect.*log(dspect));   % find the Shannon's entropy
    
    features.avmean(i, :) = avmean;
    features.std(i, :) = standard_dev;
    features.skewness(i, :) = skew;
    features.kurtosis(i, :) = kurt;
    features.activity(i, :) = activity;
    features.mobility(i, :) = mobility;
    features.complexity(i, :) = complexity;
    features.shentropy(i, :) = shentropy;
    features.spedge(i, :) = spedge;
    features.shentropyDyd(i, :) = shentropyDyd;
    
end

display('Saving...');

savedir = fullfile(feature_dir, strcat('/features_', train_or_test, patient_num,'.mat'));
save(savedir, 'features');

savedir = fullfile(feature_dir, strcat('/fileOrder_', train_or_test, patient_num,'.mat'));
save(savedir, 'file_order');

display('Mission accomplished.');

end