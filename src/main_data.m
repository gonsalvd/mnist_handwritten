ROOT_FOLDER = pwd; %need to be at top level '../proj2'
DATA_FOLDER = '/data/';
MOD_FILENAME = 'test60k-mod-imgs.mat';

readDigits=60000;
offset=0;
labelSetFile='train-labels.idx1-ubyte';
imgSetFile='train-images.idx3-ubyte';
imgFile = fullfile(ROOT_FOLDER,DATA_FOLDER,imgSetFile);
labelFile=fullfile(ROOT_FOLDER,DATA_FOLDER,labelSetFile);
[imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset);
modImages = imgs;
filenameMod = fullfile(ROOT_FOLDER,DATA_FOLDER,MOD_FILENAME);
stretchAndSave(1,imgs,filenameMod);

NUM_IMGS=size(imgs,3);
NUM_LABELS=length(labels);

tic
h=waitbar(0,'Resizing images')
for img = 1:NUM_IMGS
    showStretchPlot = 0;
    origImg = imgs(:,:,img);
    modImg = stretchImage(showStretchPlot,origImg);
    modImages(:,:,img)=modImg;
    waitbar(img/NUM_IMGS);
end
toc
