% Add LightNeuNet-MATLAB to path
if ~isdir('include')
    disp("The LightNeuNet-MATLAB library is not found.");
    disp("Run update.m to download or update the library");
    return;
end
addpath('include');
disp('The LightNeuNet-MATLAB library was loaded successfully.');