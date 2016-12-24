data0 = load('/Users/Gale/Documents/kaggle/Seizure-Prediction/data/train_1/1_2_0.mat')

figure1=figure('Position', [100, 100, 2000, 2000]);

subplot(16,1,1)
plot(data0.dataStruct.data(1:24000, 1))

subplot(16,1,2)
plot(data0.dataStruct.data(1:24000, 2))

subplot(16,1,3)
plot(data0.dataStruct.data(1:24000, 3))

subplot(16,1,4)
plot(data0.dataStruct.data(1:24000, 4))

subplot(16,1,5)
plot(data0.dataStruct.data(1:24000, 5))

subplot(16,1,6)
plot(data0.dataStruct.data(1:24000, 6))

subplot(16,1,7)
plot(data0.dataStruct.data(1:24000, 7))

subplot(16,1,8)
plot(data0.dataStruct.data(1:24000, 8))

subplot(16,1,9)
plot(data0.dataStruct.data(1:24000, 9))

subplot(16,1,10)
plot(data0.dataStruct.data(1:24000, 10))

subplot(16,1,11)
plot(data0.dataStruct.data(1:24000, 11))

subplot(16,1,12)
plot(data0.dataStruct.data(1:24000, 12))

subplot(16,1,13)
plot(data0.dataStruct.data(1:24000, 13))

subplot(16,1,14)
plot(data0.dataStruct.data(1:24000, 14))

subplot(16,1,15)
plot(data0.dataStruct.data(1:24000, 15))

subplot(16,1,16)
plot(data0.dataStruct.data(1:24000, 16))
