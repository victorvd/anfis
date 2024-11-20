% https://github.com/yoBoyio/Fuzzy-Logic-IRIS/blob/main/project.m

%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/
%                              Main Script
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

for i = 1:5
    try
        clear
        close all
        clc
        
        labels = 6;
        clusters_m = [2,3,4,5]; %to test different values
        
        global results;
        
        %=================================Read Dataset=================================
        data = readmatrix('D:\Documents\MATLAB\Fuzzy_Projects\vran\vranf.csv');
        %Article: Open Radio Access Networks (O-RAN) Experimentation Platform: Design and Datasets
        
        %data(:,22:26) = [];%labels depending
        %data(:,21) = [];
        
        % Cross varidation (train: 80%, test: 20%)
        inst = size(data,1);
    
        data = data(randperm(inst,inst),:) ;  % randomise the data
        testing_set = data(round(0.8*inst)+1:end,:) ;  % 20% take test continuously
        
        data = data(1:round(0.8*inst),:) ;  % pick the left data 80%
        
        inst = size(data,1);
         
        data = data(randperm(inst,inst),:) ;  % randomise the data 
        training_set = data(1:round(0.75*inst),:) ;
        validation_set = data(round(0.75*inst)+1:end,:) ;
        
        %==================Initialize Table to store numerical results=================
        
        table_head = {'Method','Successful Prediction','Success Rate','Total Rules'};
        results = table();
        
        epochs = 5;
        
        clusters = 5;
        fuzzycm(training_set,validation_set,testing_set,labels,epochs,clusters,table_head);
        
        results
        
        %==============================================================================
        break;  % Break out of the i-loop on success
    
    catch ME
        disp(ME);
        fprintf('Retrying...\n');
    end
end


%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/
%                          Functions Definition
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

function fuzzycm(trainset,valset,testset,labels,epochs,clusters,table_head)

    row_name = "FCM";

    features = size(trainset,2);
    
    traindat = trainset(:,1:features-labels);
    valdat = valset(:,1:features-labels);
    testdat = testset(:,1:features-labels);

    for  n = 1 : labels
        
        traintarg = trainset(:,features-labels+n);
        valtarg = valset(:,features-labels+n);
        testtarg = testset(:,features-labels+n);
        
        fis = fuzzycm_fis(traindat,traintarg,clusters)        
        fis = train_neuron(fis,traindat,traintarg,epochs);  %anfis algorithm

        evaluate_algorithm(fis,traindat,traintarg,valdat,valtarg,testdat,testtarg,row_name,table_head,epochs,n);
    end
    
    %evaluate_overall(valtarg,table_head);
    
%     output
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

%Defining initial fuzzy model using fuzzy c-means
function fis = fuzzycm_fis(traindat,traintarg,clusters)

    opts = genfisOptions('FCMClustering');

    % Clusters as many as membershipfuncions
    opts.NumClusters = clusters;
    opts.Verbose = true;

    fis = genfis(traindat,traintarg,opts)
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

function evaluate_algorithm(fis,traindat,traintarg,valdat,valtarg,testdat,testtarg,row_title,table_head,epochs,col)
global results;
    
    % Output results in testing data
    % evalfis ≡ predict
    learnOut(:,col) = evalfis(fis,valdat);
    
    gralOut(:,col) = evalfis(fis,testdat);

    [learnSuccess,learnRate] = get_success_results(valtarg,learnOut(:,col));
    [gralSuccess,gralRate] = get_success_results(testtarg,gralOut(:,col));

    % format 
    rules = showrule(fis);
    total_rules = size(rules,1);
    success_results = cell2table({row_title,learnSuccess,learnRate,total_rules},'VariableNames',table_head);
    gralSx_results = cell2table({"GENER",gralSuccess,gralRate,total_rules},'VariableNames',table_head);

    % Saving numerical results
    results = [results;success_results;gralSx_results];
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

function evaluate_overall(testtarg,table_head)
global results;    
    [row, col] = size (testtarg);

    output = zeros([row col]);

    rMatch = 0;
    
    for  i = 1 : row
        cMatch = 0;
        for  j = 1 : col
            if ((output(i,j) - testtarg(i,j)) <= 0.03)
                cMatch = cMatch + 1;
            end
        end
        if (cMatch >= col)
            rMatch = rMatch + 1;
        end
    end
    
    success_rate = sprintf("%d %%",round((rMatch/row)*100));
    success_results = cell2table({"OVERALL",rMatch,success_rate,2*6},'VariableNames',table_head);
    
    results = [results;success_results];
    
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/

%train neuron
function trainfis = train_neuron(fis,traindat,traintarg,epochs)

    % Neuron Training
    % trainfis = anfis([training_data training_targets(:,col)], fis, epochs, [], [testing_data testing_targets(:,col)]);

    %--//--//Different method--//--//
    [in,out,rule] = getTunableSettings(fis);
    opt = tunefisOptions("Method","anfis","OptimizationType","tuning");
    opt.MethodOptions.EpochNumber = epochs;	%Anfis parameter
    opt.Display = "none";
    % opt.MethodOptions.MaxGenerations = epochs;	%Genetic parameter
    % opt.MethodOptions.MaxIterations = epochs;     %Particleswarm parameter

    trainfis = tunefis(fis,[in;out],traindat,traintarg,opt);

    opt
    %--//--//--//--//--//--//--//--//
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/


%success rate
function [successful,success_rate] = get_success_results(testtarg,output)

    % Finding successful ones
    %successful = size(find((output == testing_targets) == 1), 1);
    successful = size(find((abs(output - testtarg)) <= 0.03), 1);
    %media = [((output - testing_targets) ./ testing_targets)];
    media = rmse(output,testtarg);

    % Success rate
    success_rate = sprintf("%.6f %%",media);
    %success_rate = sprintf("%d %%",100 - round(mean(media)*100));
end
%==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/==/
