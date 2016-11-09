clear classes;
clear;
clc;
close all;

% you should have python installed
pyversion;

% adds current folder to MATLAB's python search path (kludge: current
% folder must contain langModelMod)
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% Reload python module
mod = py.importlib.import_module('f1');
py.reload(mod)

% Generate dummy data
x=[[0,0,1];[1,1,0];[1,1,1];[0,0,0];[0,1,0]];
y=[[1,1,1];[1,1,0];[1,1,1];[0,1,0];[1,1,0]];

percentage = 0.1;


% Transfer data. Python function prints so we get a sense that the output
% is laid out the same
outputCell = py.f1.scoring(toggleNumpy(x), toggleNumpy(y), percentage);


