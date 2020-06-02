%先下好image process toolbox，前100行主要是图像预处理
classdef Trainer < matlab.System
    properties(Constant)
        imgSize = [320,320];
        objSize = [40,40];
        defBlockSize = [8 8];
        defNoiseThreshold = 120*120;    %噪声阈值
    end

    methods(Access = public)
        function enhancedBinaryImg = imenhance(self, rawImgPath, noiseThreshold)
            img = imread(rawImgPath);
            % 上过图像处理应该都能看得懂
            if size(img,3)==3 %彩色图片的话，就先转成灰度图片
                img = rgb2gray(img);
            end
            img = imbinarize(img);
            img = ~img; 
            %%%%图像阈值化处理
            if nargin < 3
                noiseThreshold = self.defNoiseThreshold;
            end
            enhancedBinaryImg = bwareaopen(img, noiseThreshold);
        end
        
        function [imgObjects, rectPositions] = extractObjects(self, enhancedBinaryImg)  
            %图像分割
            objects = bwconncomp(enhancedBinaryImg,8);
            %对于每个提取的对象，将imgObjects初始化为单元格数组
            imgObjects = cell(objects.NumObjects, 1);
            rectPositions = cell(objects.NumObjects, 1);
            parfor obj=1:objects.NumObjects 
                coloredPixelsIdx = objects.PixelIdxList(1,obj);
                objImg = false(objects.ImageSize);
                for i=1:numel(coloredPixelsIdx)
                    objImg(coloredPixelsIdx{i}) = 1;
                end
                
                %边界获取
                s = regionprops(objImg, 'BoundingBox');
                rect = s.BoundingBox;
                %裁剪并调整提取对象的大小，然后添加到imgObjects数组
                imgObjects{obj} = imresize(imcrop(objImg, rect), self.objSize);
                rectPositions{obj} = rect;
            end        
        end
        
        function Centroid = getCentroid(~, imgObject)
            [m,n] = size(imgObject);
            X_hist=sum(imgObject,1); 
            Y_hist=sum(imgObject,2); 
            X=1:n; Y=1:m;
            if sum(X_hist) == 0
                centX = 0;
            else
                centX=sum(X.*X_hist)/sum(X_hist); 
            end
            if sum(Y_hist) == 0
                centY = 0;
            else
                centY=sum(Y'.*Y_hist)/sum(Y_hist);
            end
            Centroid = [centX centY];
        end%特征计算：图像质心
        
        function Medoid = getMedoid(~, imgObject)
            imgObject = double(imgObject); 
            logical_med = imgObject==median(imgObject(:));
            med_indexes = find(logical_med);
            if numel(med_indexes) == 0
                logical_med = ~logical_med;
                med_indexes = find(logical_med);
            end
            [X,Y] = find(logical_med);
            med_val = med_indexes(round(numel(med_indexes)/2));
            med_index = find(med_indexes==med_val);
            medX = X(med_index);
            medY = Y(med_index);
            Medoid = [medX, medY];
          
        end%特征计算：图像中心
        
        function Perimeter = getPerimeter(~, imgObject)%特征计算：周长
            I=zeros(size(imgObject)); 
            I(2:end-1,2:end-1)=1;
            Perimeter = sum(reshape(imgObject.*I,1,[]));
        end
        
        function Area = getArea(~, imgObject)
            Area = 0;
            for i=1:numel(imgObject)
                if(imgObject(i))
                    Area = Area + 1;
                end
            end
        end%面积
       %%%%%%训练 
        function [dataSet, dataSetClasses, rectPositions] = Train(self, dataClasses, imagePaths2D, noiseThreshold, blockSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    if nargin < 4
                        enhancedBinImg = self.imenhance(curImgPath);
                    else
                        enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    end
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        if nargin < 5
                            curObjSegms = self.segment(curObj);
                        else
                            curObjSegms = self.segment(curObj, blockSize);
                        end                   
                        numOfFeatures = 11;
                        if ~dataSet_Initialized %第一次需要初始化
                            dataSet = zeros(0, numel(curObjSegms)*numOfFeatures);
                            dataSet_Initialized = 1;
                        end
                        colRange = 1:numOfFeatures;
                        [m,~] = size(dataSet);
                        %这里是将一个curObjsem作为一个处理的对象，也就是经过分割及相关处理之后的小图像
                        for segIdx = 1:numel(curObjSegms)
                            curObjSegm = curObjSegms{segIdx};
                            featureVector = zeros(1, numOfFeatures);
                            %将所有特征数据存入特征数组
                            featureVector(1,1:2) = self.getCentroid(curObjSegm);
                            featureVector(1,3:4) = self.getMedoid(curObjSegm);
                            featureVector(1,5) = self.getPerimeter(curObjSegm);
                            featureVector(1,6) = self.getArea(curObjSegm);
                            s = regionprops(curObjSegm,'Euler');
                            try
                                featureVector(1,7) = s.EulerNumber;
                            catch
                                featureVector(1,7) = 0;
                            end
                            s = regionprops(curObjSegm,'Extent');
                            try
                                featureVector(1,8) = s.Extent;
                            catch
                                featureVector(1,8) = 0;
                            end
                            s = regionprops(curObjSegm,'MajorAxisLength');
                            try
                                featureVector(1,9) = s.MajorAxisLength;                            
                            catch
                                featureVector(1,9) = 0;
                            end
                            s = regionprops(curObjSegm,'MinorAxisLength');
                            try
                                featureVector(1,10) = s.MinorAxisLength;
                            catch
                                featureVector(1,10) = 0;
                            end
                            s = regionprops(curObjSegm,'Orientation');
                            try
                                featureVector(1,11) = s.Orientation;
                            catch
                                featureVector(1,11) = 0;
                            end
                            % 将特征向量存储入dataSet
                            if max(featureVector)
                                featureVector = abs(featureVector);
                                featureVector = featureVector/max(featureVector); %归一化
                            end
                            dataSet(m+1, colRange) = featureVector;
                            
                            colRange = colRange + numOfFeatures;
                        end
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
        %将图像分割到defBlocksize大小
        function imgSegments = segment(self, img, blockSize)
            [imgX,imgY] = size(img);
            if nargin < 3
                blockSize = self.defBlockSize;
            end
            blockX = blockSize(1);
            blockY = blockSize(2);
            imgSegments = cell(0, 1);
            if blockX <= imgX && blockY <= imgY
                x1 = 1; x2 = blockX;
                while x2 <= imgX
                    y1 = 1; y2 = blockY;
                    while y2 <= imgY
                        imgSegments{end+1} = img(x1:x2, y1:y2);
                        y1 = y2+1;
                        y2 = y2+blockY;
                        
                        if y1 <= imgY && y2 > imgY
                            y1 = y1-(y2-imgY);
                            y2 = imgY;
                        end
                        
                    end
                    x1 = x2+1;
                    x2 = x2+blockX;
                    
                    if x1 <= imgX && x2 > imgX
                        x1 = x1-(x2-imgX);
                        x2 = imgX;
                    end
                    
                end
            else
                imgSegments{1} = img;
            end
        end
        %%%%HOG（方向梯度直方图）训练，。，。，。
        function [dataSet, dataSetClasses, rectPositions] = TrainHOG(self, dataClasses, imagePaths2D, noiseThreshold, CellSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    if nargin < 4
                        enhancedBinImg = self.imenhance(curImgPath);
                    else
                        enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    end 
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        if nargin < 5
                            CellSize = self.defBlockSize;
                        end
                        hogFeatures = extractHOGFeatures(curObj,'CellSize', CellSize);
                        if ~dataSet_Initialized
                            dataSet = zeros(0, numel(hogFeatures));
                            dataSet_Initialized = 1;
                        end
                        dataSet(end+1,:) = hogFeatures;
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = TrainAsync(self, dataClasses, imagePaths2D, noiseThreshold, blockSize, isHOG)
            switch nargin
                case 5
                    isHOG = 1;
                case 4
                    isHOG = 1;
                    blockSize = self.defBlockSize;
                case 3
                    isHOG = 1;
                    blockSize = self.defBlockSize;
                    noiseThreshold = self.defNoiseThreshold;
            end
            if isHOG
                parfor classIdx = 1:numel(dataClasses)
                    [dataSetCell{classIdx,1}, dataSetClassesCell{classIdx,1}, rectPositionsCell{classIdx,1}] = self.TrainHOG({dataClasses{classIdx}}, {imagePaths2D{classIdx}}, noiseThreshold, blockSize);
                end
            else
                parfor classIdx = 1:numel(dataClasses)
                    [dataSetCell{classIdx,1}, dataSetClassesCell{classIdx,1}, rectPositionsCell{classIdx,1}] = self.Train({dataClasses{classIdx}}, {imagePaths2D{classIdx}}, noiseThreshold, blockSize);
                end
            end
           
            dataSet = zeros(0, numel(dataSetCell{1,1}(1,:)));
            dataSetClasses = cell(0, 1);
            rectPositions = cell(0, 1);
            parfor classIdx = 1:numel(dataClasses)
                dataSet = vertcat(dataSet, dataSetCell{classIdx,1});
                dataSetClasses = vertcat(dataSetClasses, dataSetClassesCell{classIdx,1});
                rectPositions = vertcat(rectPositions, rectPositionsCell{classIdx,1});
            end
        end
    end
    
    methods(Access = private)
        function [MedoidVal, MedoidIdx] = extractMedoidRow(~, imgObject)
            [m,n] = size(imgObject);
            MedoidVal = zeros(1,n);
            lastRowDistance = 0;
            for r = 1:m
                rowDistance = 0;
                for c = 1:n
                    curEl = imgObject(r,c);
                    for internalRow = 1:m
                        rowDistance = rowDistance + abs(curEl - imgObject(internalRow, c));
                    end
                end

                if r == 1
                    MedoidVal = imgObject(1,:); 
                    MedoidIdx = 1;
                    lastRowDistance = rowDistance;
                else
                    if rowDistance < lastRowDistance
                        MedoidVal = imgObject(r,:);
                        MedoidIdx = r;
                        lastRowDistance = rowDistance;
                    end
                end
            end
        end
    end
end
