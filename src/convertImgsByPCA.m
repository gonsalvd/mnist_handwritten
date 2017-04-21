function convertImgsByPCA(imgs,mu,coeff)
%% Function - convert images to PCA subspace
% imgs - S X S X N, S-dimension of img (e.g. 20x20), N-num samples
% mu - size S^2 determined from pca()
% coeff - size S^2xS^2 determined using pca() from some sample of images

DIM_SIZE = size(imgs,1);
NUM_IMGS = size(imgs,3);

imgs = transpose(reshape(imgs,DIM_SIZE*DIM_SIZE,NUM_IMGS)); %reshapes to NxM, N-samples, M=S^2

%transform
h = waitbar(0,'Transform to PCA subspace...');
for i = 1:NUM_IMGS
    sample = imgs(i,:); 
    sample = sample - mu;   %center to learned PCA data
    sample = sample * coeff; %features of PCA subspace
    imgs(i,:) = sample';
    waitbar(i/NUM_IMGS);
end

close(h);
%save?

end
    
