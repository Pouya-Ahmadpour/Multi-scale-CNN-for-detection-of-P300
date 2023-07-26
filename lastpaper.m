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
meannum = 15;
datanum = 2550/meannum;
for tens = 1:datanum

    tensor1{1}(:,:,1,2*tens-1) = mean(s1nontargets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{1}(2*tens-1,1) = 0;
    tensor1{1}(:,:,1,2*tens) = mean(s1targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{1}(2*tens,1) = 1;
    
        tensor1{2}(:,:,1,2*tens-1) = mean(s1nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+datanum),3);
    labeltensor1{2}(2*tens-1,1) = 0;
    tensor1{2}(:,:,1,2*tens) = mean(s1targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{2}(2*tens,1) = 1;
    
        tensor1{3}(:,:,1,2*tens-1) = mean(s1nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+2*datanum),3);
    labeltensor1{3}(2*tens-1,1) = 0;
    tensor1{3}(:,:,1,2*tens) = mean(s1targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{3}(2*tens,1) = 1;
    
        tensor1{4}(:,:,1,2*tens-1) = mean(s1nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+3*datanum),3);
    labeltensor1{4}(2*tens-1,1) = 0;
    tensor1{4}(:,:,1,2*tens) = mean(s1targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{4}(2*tens,1) = 1;
    
        tensor1{5}(:,:,1,2*tens-1) = mean(s1nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+4*datanum),3);
    labeltensor1{5}(2*tens-1,1) = 0;
    tensor1{5}(:,:,1,2*tens) = mean(s1targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor1{5}(2*tens,1) = 1;
        
end
%
datanum = 1200/meannum;
for tens = 1:datanum

    tensor2{1}(:,:,1,2*tens-1) = mean(s2nontargets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{1}(2*tens-1,1) = 0;
    tensor2{1}(:,:,1,2*tens) = mean(s2targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{1}(2*tens,1) = 1;
    
        tensor2{2}(:,:,1,2*tens-1) = mean(s2nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+datanum),3);
    labeltensor2{2}(2*tens-1,1) = 0;
    tensor2{2}(:,:,1,2*tens) = mean(s2targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{2}(2*tens,1) = 1;
    
        tensor2{3}(:,:,1,2*tens-1) = mean(s2nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+2*datanum),3);
    labeltensor2{3}(2*tens-1,1) = 0;
    tensor2{3}(:,:,1,2*tens) = mean(s2targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{3}(2*tens,1) = 1;
    
        tensor2{4}(:,:,1,2*tens-1) = mean(s2nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+3*datanum),3);
    labeltensor2{4}(2*tens-1,1) = 0;
    tensor2{4}(:,:,1,2*tens) = mean(s2targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{4}(2*tens,1) = 1;
    
        tensor2{5}(:,:,1,2*tens-1) = mean(s2nontargets(:,:,((tens*meannum-(meannum-1)):tens*meannum)+4*datanum),3);
    labeltensor2{5}(2*tens-1,1) = 0;
    tensor2{5}(:,:,1,2*tens) = mean(s2targets(:,:,(tens*meannum-(meannum-1)):tens*meannum),3);
    labeltensor2{5}(2*tens,1) = 1;

end
%}
%% save and load tensor
%save('C:\Users\pouya\Desktop\dataset 2\tensor1.mat', 'tensor1', '-v7.3');
%save('C:\Users\pouya\Desktop\dataset 2\tensor2.mat', 'tensor2', '-v7.3');
%save('C:\Users\pouya\Desktop\dataset 2\labeltensor1.mat', 'labeltensor1');
%save('C:\Users\pouya\Desktop\dataset 2\labeltensor2.mat', 'labeltensor2');
%{
load('C:\Users\pouya\Desktop\dataset 2\tensor1.mat');
load('C:\Users\pouya\Desktop\dataset 2\tensor2.mat');
load('C:\Users\pouya\Desktop\dataset 2\labeltensor1.mat');
load('C:\Users\pouya\Desktop\dataset 2\labeltensor2.mat');

%}


%% test data
%
k = 1;
    for epochs = 41:85
        for wind = 1:180
            for chan = 1:64
            testedited(chan,:,1,k) = train(2,epochs,wind,:,chan);
            end
            test_yedited(k,1) = train_y(2,epochs,wind);
            testvalueedited(k,1) = dataTrain{2}.StimulusCode(epochs,windowofanalysis(wind,1));
            k = k+1;
        end
    end
%}
%% CNN training
%
for inputnum = 1
     sizes = [64 160 1];
%
     layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input')
        convolution2dLayer([64,10],16,'Stride',[1,10],'WeightsInitializer','ones','Name','conv_1')
        reluLayer('Name','relu_1')
        dropoutLayer(0.25,'Name','d1')
        fullyConnectedLayer(2,'WeightsInitializer','narrow-normal','Name','FC1')
        softmaxLayer('Name','softmax2')
        classificationLayer('Name','classifier')
        ];
    %figure
    %plot(lgraph)
  %
    options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',30,...
        'MiniBatchSize',round(100/meannum), ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.0005,...
        'ExecutionEnvironment','cpu');
    % 'Plots','training-progress'
    label1{inputnum} = categorical(labeltensor1{inputnum});
main_net = trainNetwork(tensor1{inputnum},label1{inputnum},layers,options);

sv_name=['C:\Users\pouya\Desktop\dataset 2\main_net_',int2str(inputnum),'.mat']; 
    save(sv_name,'main_net');
%main(inputnum) = load(sv_name);

end
%}
%% laod networks
%
main(1) = load('C:\Users\pouya\Desktop\dataset 2\main_net_1.mat');
main(2) = load('C:\Users\pouya\Desktop\dataset 2\main_net_2.mat');
main(3) = load('C:\Users\pouya\Desktop\dataset 2\main_net_3.mat');
main(4) = load('C:\Users\pouya\Desktop\dataset 2\main_net_4.mat');
main(5) = load('C:\Users\pouya\Desktop\dataset 2\main_net_5.mat');

%}
%% fine tuning
%
for inputnum = 1
    %
    sizes = [64 160 1];
layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input')
        convolution2dLayer([64,10],16,'Stride',[1,10],'WeightsInitializer','ones','Name','conv_1')
        reluLayer('Name','relu_1')
        dropoutLayer(0.25,'Name','d1')
        fullyConnectedLayer(2,'WeightsInitializer','narrow-normal','Name','FC1')
        softmaxLayer('Name','softmax2')
        classificationLayer('Name','classifier')
        ];
    layers(2, 1).Weights = main(inputnum).main_net.Layers(2, 1).Weights;
    layers(5, 1).Weights = main(inputnum).main_net.Layers(5, 1).Weights;

        %figure
        %plot(lgraph)
    %
     options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',100,...
        'MiniBatchSize',round(100/meannum), ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.0005,...
        'ExecutionEnvironment','cpu');
      label2{inputnum} = categorical(labeltensor2{inputnum});
finetuned_net = trainNetwork(tensor2{inputnum},label2{inputnum},layers,options);
sv_name=['C:\Users\pouya\Desktop\dataset 2\finetuned_net_',int2str(inputnum),'.mat']; 
    save(sv_name,'finetuned_net');
end

%% load up fine tuned CNNs
main2(1) = load('C:\Users\pouya\Desktop\dataset 2\finetuned_net_1.mat');
main2(2) = load('C:\Users\pouya\Desktop\dataset 2\finetuned_net_2.mat');
main2(3) = load('C:\Users\pouya\Desktop\dataset 2\finetuned_net_3.mat');
main2(4) = load('C:\Users\pouya\Desktop\dataset 2\finetuned_net_4.mat');
main2(5) = load('C:\Users\pouya\Desktop\dataset 2\finetuned_net_5.mat');

%}
%% svm training without feature selection
%{
for inputnum = 1:5
featuresTrain{inputnum} = activations(main2(inputnum).finetuned_net,tensor1{inputnum},"FC2");
featuresTrain{inputnum} = squeeze(featuresTrain{inputnum});
featuresTrain{inputnum} = featuresTrain{inputnum}';
mdl = fitcsvm(featuresTrain{inputnum},labeltensor1{inputnum});

featuresTrain{inputnum} = activations(main2(inputnum).finetuned_net,tensor2{inputnum},"FC2");
featuresTrain{inputnum} = squeeze(featuresTrain{inputnum});
featuresTrain{inputnum} = featuresTrain{inputnum}';
mdl = fitcsvm(featuresTrain{inputnum},labeltensor2{inputnum});



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
%
class(:,1) = classify(main2(1).finetuned_net,testedited);
class(:,2) = classify(main2(2).finetuned_net,testedited);
class(:,3) = classify(main2(3).finetuned_net,testedited);
class(:,4) = classify(main2(4).finetuned_net,testedited);
class(:,5) = classify(main2(5).finetuned_net,testedited);

class = string(class);
class = double(class);
for te = 1:8100
myclass(te,1) = mode(class(te,:));
end
accuracy = 100*sum(myclass == test_yedited)/8100
%}
%% feature selection

epochnum = meannum;
acc = 0;
clear class
clear classmain
clear predicted
for char = 1:45
   clear class
clear classmain
clear predicted 
dataextractor = testedited(:,:,1,(((char-1)*180)+1):char*180);    
codeextractor = testvalueedited((((char-1)*180)+1):char*180,1);
labelextractor = test_yedited((((char-1)*180)+1):char*180,1);
% to find classes
labelex = labelextractor(1:12,1);
codeex = codeextractor(1:12,1);
labelsext = find(labelex == 1);
realclass = codeextractor(labelsext);

coder = codeextractor(1:12*epochnum);

for ex = 1:12
    t = 1;
    clear datacaollector
    for mine = 1:length(coder)
        if coder(mine,1) == ex
            datacollector(:,:,1,t) = dataextractor(:,:,1,t);
            t = t+1;
        end
    end
    meaner(:,:,1,ex) = mean(datacollector,4);
end
    
class(:,1) = classify(main2(1).finetuned_net,meaner);
class(:,2) = classify(main2(2).finetuned_net,meaner);
class(:,3) = classify(main2(3).finetuned_net,meaner);
class(:,4) = classify(main2(4).finetuned_net,meaner);
class(:,5) = classify(main2(5).finetuned_net,meaner);
%}
%{
featuresTest1 = activations(main2(1).finetuned_net,meaner(:,:,1,:),"FC2");
featuresTest1 = squeeze(featuresTest1);
featuresTest1 = featuresTest1';

featuresTest2 = activations(main2(2).finetuned_net,meaner(:,:,1,:),"FC2");
featuresTest2 = squeeze(featuresTest2);
featuresTest2 = featuresTest2';

featuresTest3 = activations(main2(3).finetuned_net,meaner(:,:,1,:),"FC2");
featuresTest3 = squeeze(featuresTest3);
featuresTest3 = featuresTest3';

featuresTest4 = activations(main2(4).finetuned_net,meaner(:,:,1,:),"FC2");
featuresTest4 = squeeze(featuresTest4);
featuresTest4 = featuresTest4';

featuresTest5 = activations(main2(5).finetuned_net,meaner(:,:,1,:),"FC2");
featuresTest5 = squeeze(featuresTest5);
featuresTest5 = featuresTest5';


class(:,1) = predict(model(1).mdl,featuresTest1);
class(:,2) = predict(model(2).mdl,featuresTest2);
class(:,3) = predict(model(3).mdl,featuresTest3);
class(:,4) = predict(model(4).mdl,featuresTest4);
class(:,5) = predict(model(5).mdl,featuresTest5);
%}
class = string(class);
class = str2double(class);
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
ac = mean(accchar)
for g = 1:5
    h(g,:) = prediction{g};
    hh(g,:) = real{g};
end
%}
%}

%% test without no feature selection
%{
acc = 0;
clear class
clear classmain
clear ber
clear coder
clear codepos
clear meaner
epochnumber = 15;
for char = 1
    
dataextractor = testedited(:,:,1,(((char-1)*180)+1):char*180);    
codeextractor = testvalueedited((((char-1)*180)+1):char*180,1);
labelextractor = test_yedited((((char-1)*180)+1):char*180,1);
labelex = labelextractor(1:12,1);
codeex = codeextractor(1:12,1);
labelsext = find(labelex == 1);
realclass = codeextractor(labelsext);
ber(1,:) = 1:12*epochnumber;
%ber(2,:) = 12*epochnumber+1:12*epochnumber*2;
%ber(3,:) = 12*epochnumber*2+1:12*epochnumber*3;
%ber(4,:) = 12*epochnumber*3+1:12*epochnumber*4;
%ber(5,:) = 12*epochnumber*4+1:12*epochnumber*5;
ber = ber';
for epoch = 1:15/epochnumber
coder = codeextractor(ber(:,epoch));
for ex = 1:12
codepos = find(coder == ex);
meaner(:,:,1,ex) = mean(dataextractor(:,:,1,codepos),4);
end
%
class(:,epoch) = classify(main2(1).finetuned_net,meaner);
end
for mod = 1:length(class)
classmain(mod,1) = mode(class(mod,:));
end

end
%}
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
filter_order=2;
low_cutoff = 0.1;
high_cutoff = 20;
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
dataTrain{trainingdata}.StimulusCode = double(dataTrain{trainingdata}.StimulusCode);
dataTrain{trainingdata}.StimulusType = double(dataTrain{trainingdata}.StimulusType);
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