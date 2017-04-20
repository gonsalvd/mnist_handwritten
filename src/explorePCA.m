function explorePCA(doPCA, imgs)

if(doPCA)
    DIM_SIZE = size(imgs,1);
    NUM_IMGS = size(imgs,3);
    NUM_SELECT = 10000; %grab num_select chunks of random samples and
    num_trials = 10;
    %figure;
    
    for t = 1:num_trials
        %% -----SELECT DATA-----
        randSelection = randperm(NUM_IMGS,NUM_SELECT);
        randImgs = imgs(:,:,randSelection);
        trialSubset = transpose(reshape(randImgs,DIM_SIZE*DIM_SIZE,NUM_SELECT));
        
        %% --------PCA-------
        [coeff,score,latent,tsquared,explained,mu] = pca(trialSubset);
        
        %% -------PLOT------
        %figure(1066);
        subplot(2,2,1);
        hold on;
        plot(explained);
        xlabel('Princ Comp');
        ylabel('Perc Variance Expl');
        title('Ind - Percent of Variance Explained');
        subplot(2,2,2);
        hold on;
        plot(cumsum(explained));
        xlabel('Princ Comp');
        ylabel('Total Perc Variance Expl');
        title('Total - Percent of Variance Explained');
        subplot(2,2,3);
        hold on;
        plot(mu);
        xlabel('Orig Variable');
        ylabel('Mu');
        title('Mus over samples');
    end
end

end