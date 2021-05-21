addpath('data');
product = 'AUDUSD';

data = readtable('USDJPY.csv');
interval = 2048;

tic
qplTimeSeries = zeros((size(data, 1)-interval), 1);

% Sliding window
for i=1:(size(data, 1)-interval)
    qpls = calculateQPL(data(i:(i+interval)-1, :), interval);
    qplTimeSeries(i) = qpls(21, 2);
end
toc

out = data(1:(size(data, 1)-interval), 1:3);
out{:, 4} = qplTimeSeries;
out.Properties.VariableNames{4} = 'QPL';

writetable(out, './out/USDJPY_QPL.csv', 'Delimiter', ',')