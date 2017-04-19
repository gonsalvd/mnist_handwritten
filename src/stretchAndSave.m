function stretchAndSave(doFunc,imgs,SAVE_FILE)
%% Function - Stretch and save all images.
% imgs - all images
% doFunc - do this function

if(~doFunc)
    return
end

NUM_IMGS=size(imgs,3);
SIZE_IMG = size(imgs(:,:,1),1);
showStretchPlot = 0;

modImages = zeros(SIZE_IMG,SIZE_IMG,NUM_IMGS);

tic
h=waitbar(0,'Resizing images and saving...')
for img = 1:NUM_IMGS
    origImg = imgs(:,:,img);
    modImg = stretchImage(showStretchPlot,origImg);
    modImages(:,:,img)=modImg;
    waitbar(img/NUM_IMGS);
end
disp(sprintf('Took %.1f seconds to resize.',toc));
close(h);

if(~exist(SAVE_FILE,'file'))
    imgs = modImages;
    save(SAVE_FILE,'imgs');
else
    disp('Do you want to re-save? Save not done');
end

end