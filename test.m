%clear all;
%你请提前跑好结果出来，我跑了蛮久的，当然也可能是我电脑性能不太好
clc;
[y,Fs] = audioread('done.wav');
ir = ImageReader;
cl = Classifier;

%下面这些当前文件夹下数据可以选择

%load('trainHOG_4x4_Cells.mat');
%load('trainHOG_8x8_Cells.mat');
%load('train_4x4_Blocks.mat');
%load('train_8x8_Blocks.mat');

%训练%{
[dataClasses, imagePaths2D] = ir.read('dataset');%ImageReader.m中定义的read
tic; 
[trainedSetHOG, trainedSetClassesHOG] = cl.tr.TrainHOG(dataClasses, imagePaths2D, 120*120, [8 8]);
[trainedSetHOG, trainedSetClassesHOG] = cl.tr.TrainAsync(dataClasses, imagePaths2D, 120*120, [8 8], 1); % 1 = HOG
[trainedSet, trainedSetClasses] = cl.tr.Train(dataClasses, imagePaths2D, 120*120, [4 4]);
[trainedSet, trainedSetClasses] = cl.tr.TrainAsync(dataClasses, imagePaths2D, 120*120, [4 4], 0);
elapsedTrainingTimeMinutes = toc/60;%经过的训练时间
sound(y,Fs);
%}

%测试%{
tic;
[testObjectsHOG, testObjectsPosHOG] = cl.getImgReadyHOG('testImgs/test4.jpg', 10, [8 8]);
I = imread('testImgs/test4.jpg');
for i=1:numel(testObjectsHOG(:,1))
	classTypeHOG = cl.weightedKNN(trainedSetHOG, trainedSetClassesHOG, testObjectsHOG(i,:), 3, 0);
    % 显示原始图像中对象的位置
    I = insertObjectAnnotation(I, 'rectangle', testObjectsPosHOG{i}, classTypeHOG,'TextBoxOpacity',0.5,'FontSize',12);
end
imshow(imresize(I, 1));
elapsedClassificationTime = toc;
%}%{
tic; 
[testObjects, testObjectsPos] = cl.getImgReady('testImgs/test4.jpg', 10, [4 4]);
I = imread('testImgs/test4.jpg');
for i=1:numel(testObjects(:,1))
	classType = cl.weightedKNN(trainedSet, trainedSetClasses, testObjects(i,:), 3, 0);
    I = insertObjectAnnotation(I, 'rectangle', testObjectsPos{i}, classType,'TextBoxOpacity',0.5,'FontSize',12);
end
imshow(imresize(I, 1));
elapsedClassificationTime = toc;
%}


tic; 
%[testObjects, testObjectsPos] = cl.getImgReadyHOG('testImgs/test4.jpg', 10, [8 8]);



[baySet, classes, classesProps] = cl.bh.getBayesianSet(trainedSetHOG, trainedSetClassesHOG, @normc);
classesTypes = cl.bayesClassifyAsync(baySet, classes, classesProps, testObjects, @normc);


I = imread('testImgs/test3.png');
 for i=1:numel(testObjects(:,1))
Show the location of the objects in the original image
  I = insertObjectAnnotation(I, 'rectangle', testObjectsPos{i}, classesTypes{i},'TextBoxOpacity',0.5,'FontSize',12);
end
imshow(imresize(I, 1));
elapsedClassificationTime = toc;
