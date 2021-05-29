%% Environment setup
clear
clc

%% Global variables
window = 5;
qplinterval = 2048;
product = 'EURUSD';

trainstartyear = 2010;
trainstartmonth = 1;
trainstartday = 1;

trainendyear = 2019;
trainendmonth = 12;
trainendday = 31;

teststartyear = 2020;
teststartmonth = 1;
teststartday = 1;

%% Load the financial time series dataset
addpath('data');
filename = product + "_DAILY.csv";
data = readtable(filename);

%% Calculate QPLs
disp("Calculating QPLs.");
tic
qpltimeseries = zeros((size(data, 1)-qplinterval), 3);
for i=1:(size(data, 1)-qplinterval)
    qpls = calculateQPL(data(i:(i+qplinterval)-1, :), qplinterval);
    % QPL0
    qpltimeseries(i, 1) = qpls(20, 2);
    % QPL+1
    qpltimeseries(i, 2) = qpls(21, 2);
    % QPL-1
    qpltimeseries(i, 3) = qpls(22, 2);
end
toc

%% Prepare training data
disp("Preparing training data. Please wait...");
done = false;
% Do not have the training data for tomorrow
i = 2;
n = size(data, 1);
numfeatures = 9;
trainX = rand(1, numfeatures);
trainy = rand(1, 1);
testX = rand(1, numfeatures);
testy = rand(1, 1);
testsize = 0;
tic
while ~done
    if data{i, 'Year'} >= teststartyear && ...
            data{i, 'Month'} >= teststartmonth && ...
            data{i, 'Day'} >= teststartday
        for j=0:(window-1)
            testX(i-1, (1+j*numfeatures):(4+j*numfeatures)) = data{i+j, 4:7};
            testX(i-1, (5+j*numfeatures):(6+j*numfeatures)) = data{i+j, 12:13};
            testX(i-1, (7+j*numfeatures):(9+j*numfeatures)) = qpltimeseries(i+j, 1:3);
            testy(i-1, :) = data{i-1, 7};
        end
        testsize = testsize + 1;
    elseif data{i, 'Year'} >= trainstartyear && ...
            data{i, 'Month'} >= trainstartmonth && ...
            data{i, 'Day'} >= trainstartday && ...
            data{i, 'Year'} <= trainendyear && ...
            data{i, 'Month'} <= trainendmonth && ...
            data{i, 'Day'} <= trainendday
        for j=0:(window-1)
            trainX(i-1-testsize, (1+j*numfeatures):(4+j*numfeatures)) = data{i+j, 4:7};
            trainX(i-1-testsize, (5+j*numfeatures):(6+j*numfeatures)) = data{i+j, 12:13};
            trainX(i-1-testsize, (7+j*numfeatures):(9+j*numfeatures)) = qpltimeseries(i+j, 1:3);
            trainy(i-1-testsize, :) = data{i-1, 7};
        end
    end
    i = i + 1;
    if i > n
        done = true;
        break;
    end
    % trainX = data()
end
toc

trainX = trainX';
trainy = trainy';
testX = testX';
testy = testy';

%% Normalize
trainyoriginal = trainy;
testyoriginal = testy;
scalerX = MinMaxScaler();
scalerY = StandardScaler();
trainX = scalerX.fittransform(trainX, 2);
trainy = scalerY.fittransform(trainy, 2);
testX = scalerX.transform(testX);
testy = scalerY.transform(testy);
disp("Data normalized");

%% Model
model = SequentialModel();

model.add(InputLayer(45));

options.activation = "sigmoid";
options.usebias = true;
options.kernelinitializer = "he";

model.add(DenseLayer(32, options));
model.add(DenseLayer(24, options));
model.add(DenseLayer(16, options));
model.add(DenseLayer(8, options));

options.activation = "linear";
options.usebias = true;
options.kernelinitializer = "he";

model.add(OutputLayer(1, options));

model.compile();

%% Train
options.batchsize = 128;
options.epochs = 20000;
options.learningrate = 0.001;
options.lambd = 0.01;
options.loss = "mse";

model.fit(trainX, trainy, options);

%% Test on the training set
probs = model.predict(trainX);
J = (1/size(trainy, 2)) * sum(sum((trainy-probs).^2));
fprintf('Training set loss: %3.5f\n', J);
figure
hold on
plot(trainyoriginal)
plot(scalerY.inversetransform(probs))

%% Test on the test set
probs = model.predict(testX);
J = (1/size(testy, 2)) * sum(sum((testy-probs).^2));
fprintf('Test set lost: %3.5f\n', J);
figure
hold on
plot(testyoriginal)
plot(scalerY.inversetransform(probs))

%% Results
% Fitness: [0.75 / 5.00]
% Training set loss: 0.08059
% Test set lost: 0.00900

