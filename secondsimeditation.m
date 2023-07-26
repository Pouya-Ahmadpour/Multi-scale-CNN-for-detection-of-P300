close all
clear all
clc
addpath('C:\Users\pouya\Desktop\dataset 2');
%% initials
channels = 64;
totalsubjects = 2;
totalsessions = 5;
totalclass = 36;
trialperepoch = 12;
totalepochs = 85;
samplingrate = 240;
stimlength = 0.1;
interval = 0.075;
stimsamples = stimlength*samplingrate;
max_epochs=10;
intervalsamples = interval*samplingrate;
%% preprocessing
%
[dataTrain,dataTest,windowlabelstrain,windowofanalysis,train,train_y] = preprocess(channels,totalsubjects,totalsessions,totalclass,trialperepoch,...
    totalepochs,stimsamples,intervalsamples,samplingrate);
%}
%% target and nontarget separation
%
t = 1;
tt = 1;
    for epochs = 1:85
        for wind = 1:180
            if train_y(1,epochs,wind) == 1
                for chan = 1:64
                s1targets(chan,:,t) = train(1,epochs,wind,:,chan);
                end
                t = t+1;
            else
                for chan = 1:64
    s1nontargets(chan,:,tt) = train(1,epochs,wind,:,chan);
                end
    tt = tt+1;
            end
        end
    end
%%
t = 1;
tt = 1;
    for epochs = 1:40
        for wind = 1:180
            if train_y(2,epochs,wind) == 1
                for chan = 1:64
                s2targets(chan,:,t) = train(2,epochs,wind,:,chan);
                end
                t = t+1;
            else
                for chan = 1:64
    s2nontargets(chan,:,tt) = train(2,epochs,wind,:,chan);
                end
    tt = tt+1;
            end
        end
    end
%% save
%{
save('C:\Users\pouya\Desktop\dataset 2\s1targets.mat','s1targets');
save('C:\Users\pouya\Desktop\dataset 2\s1nontargets.mat','s1nontargets');
save('C:\Users\pouya\Desktop\dataset 2\s2targets.mat','s2targets');
save('C:\Users\pouya\Desktop\dataset 2\s2nontargets.mat','s2nontargets');


%%
%}
% load targets and nontargets
%{
load('C:\Users\pouya\Desktop\dataset 2\s1targets.mat');
load('C:\Users\pouya\Desktop\dataset 2\s1nontargets.mat');
load('C:\Users\pouya\Desktop\dataset 2\s2targets.mat');
load('C:\Users\pouya\Desktop\dataset 2\s2nontargets.mat');
%}
%%
%
for tens = 1:2*2550
    if tens <= 2550
        %
        tensor1{1}(:,:,1,tens) = s1targets(:,:,tens);
        tensor1{2}(:,:,1,tens) = s1targets(:,:,tens);
        tensor1{3}(:,:,1,tens) = s1targets(:,:,tens);
        tensor1{4}(:,:,1,tens) = s1targets(:,:,tens);
        tensor1{5}(:,:,1,tens) = s1targets(:,:,tens);
        %
        labeltensor1{1}(tens,1) = 1;
        labeltensor1{2}(tens,1) = 1;
        labeltensor1{3}(tens,1) = 1;
        labeltensor1{4}(tens,1) = 1;
        labeltensor1{5}(tens,1) = 1;
    else
        %
        tensor1{1}(:,:,1,tens) = s1nontargets(:,:,tens-length(s1targets));
        tensor1{2}(:,:,1,tens) = s1nontargets(:,:,tens);
        tensor1{3}(:,:,1,tens) = s1nontargets(:,:,length(s1targets)+tens);
        tensor1{4}(:,:,1,tens) = s1nontargets(:,:,2*length(s1targets)+tens);
        tensor1{5}(:,:,1,tens) = s1nontargets(:,:,3*length(s1targets)+tens);
        %
        labeltensor1{1}(tens,1) = 0;
        labeltensor1{2}(tens,1) = 0;
        labeltensor1{3}(tens,1) = 0;
        labeltensor1{4}(tens,1) = 0;
        labeltensor1{5}(tens,1) = 0;
    end
end
%
for tens = 1:2*1200
    if tens <= 1200
        %
        tensor2{1}(:,:,1,tens) = s2targets(:,:,tens);
        tensor2{2}(:,:,1,tens) = s2targets(:,:,tens);
        tensor2{3}(:,:,1,tens) = s2targets(:,:,tens);
        tensor2{4}(:,:,1,tens) = s2targets(:,:,tens);
        tensor2{5}(:,:,1,tens) = s2targets(:,:,tens);
        %
        labeltensor2{1}(tens,1) = 1;
        labeltensor2{2}(tens,1) = 1;
        labeltensor2{3}(tens,1) = 1;
        labeltensor2{4}(tens,1) = 1;
        labeltensor2{5}(tens,1) = 1;
    else
        %
        tensor2{1}(:,:,1,tens) = s2nontargets(:,:,tens-length(s2targets));
        tensor2{2}(:,:,1,tens) = s2nontargets(:,:,tens);
        tensor2{3}(:,:,1,tens) = s2nontargets(:,:,length(s2targets)+tens);
        tensor2{4}(:,:,1,tens) = s2nontargets(:,:,2*length(s2targets)+tens);
        tensor2{5}(:,:,1,tens) = s2nontargets(:,:,3*length(s2targets)+tens);
        %
        labeltensor2{1}(tens,1) = 0;
        labeltensor2{2}(tens,1) = 0;
        labeltensor2{3}(tens,1) = 0;
        labeltensor2{4}(tens,1) = 0;
        labeltensor2{5}(tens,1) = 0;
    end
end
%}
%% save and load tensor
%save('C:\Users\pouya\Desktop\dataset 2\tensor1.mat', 'tensor1', '-v7.3');
%save('C:\Users\pouya\Desktop\dataset 2\tensor2.mat', 'tensor2', '-v7.3');
%save('C:\Users\pouya\Desktop\dataset 2\labeltensor1.mat', 'labeltensor1');
%save('C:\Users\pouya\Desktop\dataset 2\labeltensor2.mat', 'labeltensor2');
load('C:\Users\pouya\Desktop\dataset 2\tensor1.mat');
load('C:\Users\pouya\Desktop\dataset 2\tensor2.mat');
load('C:\Users\pouya\Desktop\dataset 2\labeltensor1.mat');
load('C:\Users\pouya\Desktop\dataset 2\labeltensor2.mat');




%% test data
%
k = 1;
    for epochs = 41:85
        for wind = 1:180
            for chan = 1:64
            testedited(chan,:,1,k) = train(2,epochs,wind,:,chan);
            end
            test_yedited(k,1) = train_y(2,epochs,wind);
            k = k+1;
        end
    end
%}
%% CNN training
%
for inputnum = 1:5
     sizes = [64 160 1];
%
     layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input')
        batchNormalizationLayer('Name','BN1')
        convolution2dLayer([64,1],16,'WeightsInitializer','ones','Name','conv_1')
        convolution2dLayer([1,20],1,'Stride',[1,10],'WeightsInitializer','narrow-normal','Name','ms1');
        concatenationLayer(2,2,'Name','concat')
        batchNormalizationLayer('Name','BN2')
        reluLayer('Name','relu_1')
        dropoutLayer(0.1,'Name','d1')
        fullyConnectedLayer(256,'WeightsInitializer','narrow-normal','Name','FC1')
        dropoutLayer(0.3,'Name','d2')
        fullyConnectedLayer(128,'WeightsInitializer','narrow-normal','Name','FC2')
        fullyConnectedLayer(2,'WeightsInitializer','narrow-normal','Name','FC3')
        softmaxLayer('Name','softmax2')
        classificationLayer('Name','classifier')
        ];
    lgraph = layerGraph(layers);
    layers(3, 1).BiasLearnRateFactor=0;
    ms2 = convolution2dLayer([1,10],1,'Stride',[1,10],'WeightsInitializer','narrow-normal','Name','ms2');
    lgraph = addLayers(lgraph,ms2);
    lgraph = connectLayers(lgraph,'conv_1','ms2');
    lgraph = connectLayers(lgraph,'ms2','concat/in2');
    %figure
    %plot(lgraph)
  %
    options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',30,...
        'MiniBatchSize',100, ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.0005,...
        'ExecutionEnvironment','cpu');
    % 'Plots','training-progress'
    label1{inputnum} = categorical(labeltensor1{inputnum});
main_net = trainNetwork(tensor1{inputnum},label1{inputnum},lgraph,options);

sv_name=['C:\Users\pouya\Desktop\dataset 2\main_net_',int2str(inputnum),'.mat']; 
    save(sv_name,'main_net');
%main(inputnum) = load(sv_name);

end
%}
%% laod networks
%{
main(1) = load('C:\Users\pouya\Desktop\dataset 2\main_net_1.mat');
main(2) = load('C:\Users\pouya\Desktop\dataset 2\main_net_2.mat');
main(3) = load('C:\Users\pouya\Desktop\dataset 2\main_net_3.mat');
main(4) = load('C:\Users\pouya\Desktop\dataset 2\main_net_4.mat');
main(5) = load('C:\Users\pouya\Desktop\dataset 2\main_net_5.mat');
%}
%% fine tuning
%{
for inputnum = 1:5
    %
    sizes = [64 160 1];
layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input')
        batchNormalizationLayer('Name','BN1')
        convolution2dLayer([64,1],16,'WeightsInitializer','ones','Name','conv_1')
        convolution2dLayer([1,20],1,'Stride',[1,10],'WeightsInitializer','narrow-normal','Name','ms1');
        concatenationLayer(2,2,'Name','concat')
        batchNormalizationLayer('Name','BN2')
        reluLayer('Name','relu_1')
        dropoutLayer(0.1,'Name','d1')
        fullyConnectedLayer(256,'WeightsInitializer','narrow-normal','Name','FC1')
        dropoutLayer(0.3,'Name','d2')
        fullyConnectedLayer(128,'WeightsInitializer','narrow-normal','Name','FC2')
        fullyConnectedLayer(2,'WeightsInitializer','narrow-normal','Name','FC3')
        softmaxLayer('Name','softmax2')
        classificationLayer('Name','classifier')
        ];
    Layers(3, 1).Weights = main(inputnum).main_net.Layers(3, 1).Weights;
    Layers(4, 1).Weights = main(inputnum).main_net.Layers(4, 1).Weights;
    Layers(9, 1).Weights = main(inputnum).main_net.Layers(9, 1).Weights;
    Layers(11, 1).Weights = main(inputnum).main_net.Layers(11, 1).Weights;
    Layers(12, 1).Weights = main(inputnum).main_net.Layers(12, 1).Weights;
        Layers(3, 1).BiasLearnRateFactor=0;      
        Layers(4, 1).Bias = main(inputnum).main_net.Layers(4, 1).Bias;
        Layers(9, 1).Bias = main(inputnum).main_net.Layers(9, 1).Bias;
        Layers(11, 1).Bias = main(inputnum).main_net.Layers(11, 1).Bias;
        Layers(12, 1).Bias = main(inputnum).main_net.Layers(12, 1).Bias;   
    
    lgraph = layerGraph(layers);
    layers(3, 1).BiasLearnRateFactor=0;
    ms2 = convolution2dLayer([1,10],1,'Stride',[1,10],'WeightsInitializer','narrow-normal','Name','ms2');
    lgraph = addLayers(lgraph,ms2);
    lgraph = connectLayers(lgraph,'conv_1','ms2');
    lgraph = connectLayers(lgraph,'ms2','concat/in2');
        %figure
        %plot(lgraph)
    %
     options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',60,...
        'MiniBatchSize',100, ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.0005,...
        'ExecutionEnvironment','cpu');
      label2{inputnum} = categorical(labeltensor2{inputnum});
finetuned_net = trainNetwork(tensor2{inputnum},label2{inputnum},lgraph,options);
sv_name=['C:\Users\pouya\Desktop\dataset 2\finetuned_net_',int2str(inputnum),'.mat']; 
    save(sv_name,'finetuned_net');
end
%}
%% svm training without feature selection
%{
for inputnum = 1:5
featuresTrain{inputnum} = activations(main(inputnum).main_net,tensor{inputnum},"FC2");
featuresTrain{inputnum} = squeeze(featuresTrain{inputnum});
featuresTrain{inputnum} = featuresTrain{inputnum}';
mdl = fitcsvm(featuresTrain{inputnum},labeltensor{inputnum});

sv_name=['C:\Users\pouya\Desktop\dataset 2\mdl_',int2str(inputnum),'.mat']; 
    save(sv_name,'mdl');

end
%}
%% load not feature selected svms
%{
model(1) = load('C:\Users\pouya\Desktop\dataset 2\mdl_1.mat');
model(2) = load('C:\Users\pouya\Desktop\dataset 2\mdl_2.mat');
model(3) = load('C:\Users\pouya\Desktop\dataset 2\mdl_3.mat');
model(4) = load('C:\Users\pouya\Desktop\dataset 2\mdl_4.mat');
model(5) = load('C:\Users\pouya\Desktop\dataset 2\mdl_5.mat');
%}
%% test window recognition without feature selection
%{
featureTest1 = activations(main(1).main_net,testedited,"FC2");
featureTest2 = activations(main(2).main_net,testedited,"FC2");
featureTest3 = activations(main(3).main_net,testedited,"FC2");
featureTest4 = activations(main(4).main_net,testedited,"FC2");
featureTest5 = activations(main(5).main_net,testedited,"FC2");

featureTest1 = squeeze(featureTest1);
featureTest1 = featureTest1';
featureTest2 = squeeze(featureTest2);
featureTest2 = featureTest2';
featureTest3 = squeeze(featureTest3);
featureTest3 = featureTest3';
featureTest4 = squeeze(featureTest4);
featureTest4 = featureTest4';
featureTest5= squeeze(featureTest5);
featureTest5 = featureTest5';

class(:,1) = classify(main(1).main_net,testedited);
class(:,2) = classify(main(2).main_net,testedited);
class(:,3) = classify(main(3).main_net,testedited);
class(:,4) = classify(main(4).main_net,testedited);
class(:,5) = classify(main(5).main_net,testedited);

%{
class(:,1) = predict(model(1).mdl,featureTest1);
class(:,2) = predict(model(2).mdl,featureTest2);
class(:,3) = predict(model(3).mdl,featureTest3);
class(:,4) = predict(model(4).mdl,featureTest4);
class(:,5) = predict(model(5).mdl,featureTest5);
%}
class = string(class);
class = double(class);
for te = 1:1800
myclass(te,1) = mode(class(te,:));
end
accuracy = 100*sum(myclass == test_yedited)/1800
%}
%% feature selection
%{
for inputnum = 1:5
    featureVal = activations(main(inputnum).main_net,tensor{inputnum},"FC2");
    toselect = 4601:5000;
    featureVal = featureVal(1,1,:,toselect);
    featureVal = squeeze(featureVal);
    featureVal = featureVal';
    labelVal = labeltensor{inputnum}(toselect,1);
    N = 128;
for fr = 1:N
    mainMean = mean(featureVal(:,fr));
    mainVar = (var(featureVal(:,fr)))^2;
    for i = 1:N
        if i ~= fr
mainMean = mainMean - mean(featureVal(:,i));
mainVar = mainVar - (var(featureVal(:,i)))^2;
        end
        end
Fratios(fr,1) = (mainMean^2)/mainVar;
end
[Fratiosranked,featuresnumber{inputnum}] = sort(Fratios,'descend');    
featuresnumber{inputnum} = featuresnumber{inputnum}';
rankedfeatures{inputnum} = featureVal(:,featuresnumber{inputnum});


for cv = 1:128
    
feature = rankedfeatures{inputnum}(:,1:cv);
    % 10-fold
K = 10;
n_run = 3;
accuracy = zeros(K,n_run);
% 10_fold
for i_run=1:n_run
    indices = crossvalind('Kfold',labelVal,K);
    for i_fold = 1:K
        Val = indices==i_fold;
        train = ~Val;
        afeatureTrain = rankedfeatures{inputnum}(train,:);
        featuretoVal = rankedfeatures{inputnum}(Val,:);
        Modelcross = fitcecoc(afeatureTrain,labelVal(train));
        classofcross = predict(Modelcross, featuretoVal);
        accuracyk(i_fold,1) = 100*length(find(classofcross == labelVal(Val)))...
            /length(labelVal(Val));
    end    
       accrun(i_run,1) = mean(accuracyk);  
end
totalacc(cv) = mean(accrun);
end
[useless maxfeaturenumber(inputnum)] = max(totalacc);
end
%}
%save('C:\Users\pouya\Desktop\dataset 2\maxfeaturenumber.mat', 'maxfeaturenumber');
%save('C:\Users\pouya\Desktop\dataset 2\featuresnumber.mat', 'featuresnumber');
%load('C:\Users\pouya\Desktop\dataset 2\maxfeaturenumber.mat');
%load('C:\Users\pouya\Desktop\dataset 2\featuresnumber.mat');
%% training & testing using feature selection
%{
load('C:\Users\pouya\Desktop\dataset 2\featuresnumber.mat');
%maxfeaturenumber = [50 94 108 50 120];
maxfeaturenumber = [128 128 128 128 128];
for inputnum = 1:5
featurestoTrain = activations(main(inputnum).main_net,tensor{inputnum},"FC2");
featurestoTrain = squeeze(featurestoTrain);
featurestoTrain = featurestoTrain';
featurestoTrain = featurestoTrain(:,featuresnumber{inputnum});
myfeaturestoTrain = featurestoTrain(:,1:maxfeaturenumber(inputnum));

mdl = fitcsvm(myfeaturestoTrain,labeltensor{inputnum});

sv_name=['C:\Users\pouya\Desktop\dataset 2\mdl_',int2str(inputnum),'.mat']; 
    save(sv_name,'mdl');

end

model(1) = load('C:\Users\pouya\Desktop\dataset 2\mdl_1.mat');
model(2) = load('C:\Users\pouya\Desktop\dataset 2\mdl_2.mat');
model(3) = load('C:\Users\pouya\Desktop\dataset 2\mdl_3.mat');
model(4) = load('C:\Users\pouya\Desktop\dataset 2\mdl_4.mat');
model(5) = load('C:\Users\pouya\Desktop\dataset 2\mdl_5.mat');

for char = 1:10
featureTest1 = activations(main(1).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180),"FC2");
featureTest2 = activations(main(2).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180),"FC2");
featureTest3 = activations(main(3).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180),"FC2");
featureTest4 = activations(main(4).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180),"FC2");
featureTest5 = activations(main(5).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180),"FC2");

featureTest1 = squeeze(featureTest1);
featureTest1 = featureTest1';
featureTest2 = squeeze(featureTest2);
featureTest2 = featureTest2';
featureTest3 = squeeze(featureTest3);
featureTest3 = featureTest3';
featureTest4 = squeeze(featureTest4);
featureTest4 = featureTest4';
featureTest5= squeeze(featureTest5);
featureTest5 = featureTest5';

featureTest1 = featureTest1(:,featuresnumber{1});
featureTest2 = featureTest2(:,featuresnumber{2});
featureTest3 = featureTest3(:,featuresnumber{3});
featureTest4 = featureTest4(:,featuresnumber{4});
featureTest5 = featureTest5(:,featuresnumber{5});

featureTs1 = featureTest1(:,1:maxfeaturenumber(1));
featureTs2 = featureTest2(:,1:maxfeaturenumber(2));
featureTs3 = featureTest3(:,1:maxfeaturenumber(3));
featureTs4 = featureTest4(:,1:maxfeaturenumber(4));
featureTs5 = featureTest5(:,1:maxfeaturenumber(5));
%{
class(:,1) = predict(model(1).mdl,featureTs1);
class(:,2) = predict(model(2).mdl,featureTs2);
class(:,3) = predict(model(3).mdl,featureTs3);
class(:,4) = predict(model(4).mdl,featureTs4);
class(:,5) = predict(model(5).mdl,featureTs5);
%}
%{
class(:,1) = classify(main(1).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180));
class(:,2) = classify(main(2).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180));
class(:,3) = classify(main(3).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180));
class(:,4) = classify(main(4).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180));
class(:,5) = classify(main(5).main_net,testedited(:,:,1,(((char-1)*180)+1):char*180));
%}

code = testvalueedited((((char-1)*180)+1):char*180,1);
realtest = test_yedited((((char-1)*180)+1):char*180,1);

for mod = 1:length(class)
classmain(mod,1) = mode(class(mod,:));
if classmain(mod,1) == 1
    classmain(mod,1) = code(mod,1);
end
if realtest(mod,1) == 1
    realtest(mod,1) = code(mod,1);
end
end
realtest2 = realtest;
classmain2 = classmain;

rt = mode(nonzeros(realtest(:,1)));
realtest2(realtest2==rt) = nan;
ct = mode(nonzeros(realtest2(:,1)));

rp = mode(nonzeros(classmain(:,1)));
classmain2(classmain2==rp) = nan;
cp = mode(nonzeros(classmain2(:,1)));

prediction{char} = [rp cp];
prediction{char} = sort(prediction{char});

real{char} = [rt ct];
real{char} = sort(real{char});

if prediction{char} == real{char}
       accchar(char) = 100;
       else
           accchar(char) = 0;
       end
end   
accuracy = mean(accchar)
for g = 1:5
    h(g,:) = prediction{g};
    hh(g,:) = real{g};
end
%}
%% for testing window recognition
%{
featureTest1 = activations(main(1).main_net,test,"FC2");
featureTest2 = activations(main(2).main_net,test,"FC2");
featureTest3 = activations(main(3).main_net,test,"FC2");
featureTest4 = activations(main(4).main_net,test,"FC2");
featureTest5 = activations(main(5).main_net,test,"FC2");

featureTest1 = reshape(featureTest1,[1800 128]);
featureTest2 = reshape(featureTest2,[1800 128]);
featureTest3 = reshape(featureTest3,[1800 128]);
featureTest4 = reshape(featureTest4,[1800 128]);
featureTest5 = reshape(featureTest5,[1800 128]);

featureTest1 = featureTest1(:,featuresnumber{1});
featureTest2 = featureTest2(:,featuresnumber{2});
featureTest3 = featureTest3(:,featuresnumber{3});
featureTest4 = featureTest4(:,featuresnumber{4});
featureTest5 = featureTest5(:,featuresnumber{5});

featureTs1 = featureTest1(:,1:maxfeaturenumber(1));
featureTs2 = featureTest2(:,1:maxfeaturenumber(2));
featureTs3 = featureTest3(:,1:maxfeaturenumber(3));
featureTs4 = featureTest4(:,1:maxfeaturenumber(4));
featureTs5 = featureTest5(:,1:maxfeaturenumber(5));


class(:,1) = predict(model(1).mdl,featureTs1);
class(:,2) = predict(model(2).mdl,featureTs2);
class(:,3) = predict(model(3).mdl,featureTs3);
class(:,4) = predict(model(4).mdl,featureTs4);
class(:,5) = predict(model(5).mdl,featureTs5);

for te = 1:1800
myclass(:,1) = mode(class(te,:));
end
accuracy = 100*sum(myclass == test_y)/1800
%}
%% preprocessing function
%
function [dataTrain,dataTest,windowlabelstrain,windowofanalysis,train,train_y] = preprocess(channels,totalsubjects,totalsessions,~,trialperepoch,...
    totalepochs,stimsamples,intervalsamples,samplingrate)

% filterdesign
Ripple=1;
filter_order=8;
low_cutoff = 0.1;
high_cutoff = 10;
bpfilt = designfilt('bandpassiir','FilterOrder',filter_order, ...
        'PassBandFrequency1',low_cutoff,'PassBandFrequency2',high_cutoff,...
		'PassBandRipple',Ripple,...
		'DesignMethod','cheby1','SampleRate',samplingrate);        
% filtering and data loading
% s1 = Subject_A_Train , s2 = Subject_B_Train , s3 = Subject_A_Test , s4 = Subject_B_Test
for trainingdata = 1:totalsubjects
nameofdata=['s',num2str(trainingdata),'.mat'];
    dataTrain{trainingdata}=load(nameofdata); % Loading the subject data
dataTrain{trainingdata}.Signal = double(dataTrain{trainingdata}.Signal);
dataTrain{trainingdata}.Signal = filtfilt(bpfilt,dataTrain{trainingdata}.Signal);
end
for testdata = 3:(2*totalsubjects)
    nameofdata=['s',num2str(testdata),'.mat'];
dataTest{testdata-2} = load(nameofdata);
dataTest{testdata-2}.Signal = double(dataTest{testdata-2}.Signal);
dataTest{testdata-2}.Signal = filtfilt(bpfilt,dataTest{testdata-2}.Signal);
end

%% windowing
t = 1;
for windowsnum = 1:180
    windowofanalysis(windowsnum,:) = t:(159+t);
    t = t+stimsamples+intervalsamples;    
end
%% labeling
for subs = 1:2
for epochs = 1:length(dataTrain{subs}.StimulusCode(:,1))
    for windowsnum = 1:180
if dataTrain{subs}.StimulusType(epochs,windowofanalysis(windowsnum,1)) == 1
    windowlabelstrain(subs,epochs,windowsnum) =...
        dataTrain{subs}.StimulusCode(epochs,windowofanalysis(windowsnum,1));
else
    windowlabelstrain(subs,epochs,windowsnum) = 0;
end
    end
end
end
%% data sort
for subs = 1:2
    for epochs = 1:85;
        for wind = 1:180  
                train(subs,epochs,wind,:,:) = dataTrain{subs}.Signal(epochs,windowofanalysis(wind,:),:);
            train_y(subs,epochs,wind) = dataTrain{subs}.StimulusType(epochs,windowofanalysis(wind,1));
        end
    end
end
end
%}
%%
