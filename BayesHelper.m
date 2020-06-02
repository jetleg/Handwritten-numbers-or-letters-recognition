classdef BayesHelper
    properties(Constant)    
    end  
    methods(Access = public)
        
        
        % 把 baySet输出为matrix(m,n)其中m为class的数量，n为特征组数
        % 每个都是由均值、方差构成的两个元素的数组
        function [baySet, classes, classesProps] = getBayesianSet(self, dataSet, dataSetClasses, normX)
            if nargin > 3
                dataSet = normX(dataSet); %需要先将数据集标准化
            end
            [classes, classesProps] = self.calcClassProp(dataSetClasses);
            for classIdx=1:numel(classes)
                parfor featureIdx=1:numel(dataSet(1,:))
                    classfeatureCol = self.getClassFeatureCol(dataSet(:,featureIdx), dataSetClasses, classes{classIdx});
                    baySet(classIdx, featureIdx) = self.calcMeanVariance(classfeatureCol);
                end
            end
        end
        % 计算新模型的概率，返回与最大概率相对应的类型。。。。。
        function [classType] = bayesClassify(self, baySet, classes, classesProps, newPattern)
            liklihoods = ones(size(classes));%likelihoods拼错了，你想改也可以改一下
            parfor classIdx=1:numel(classes)
                for featureIdx=1:numel(baySet(1,:))
                    s = baySet(classIdx, featureIdx);
                    x = newPattern(featureIdx);
                    if liklihoods(classIdx) == 0
                        liklihoods(classIdx) = 0.0001;%先初始化参数，
                    end
                    liklihoods(classIdx) = liklihoods(classIdx)*self.calcPartLiklihood(s.Mean, s.Variance, x);
                end
            end
            liklihoods = liklihoods.*cell2mat(classesProps);
            [~, maxClassLikIdx] = max(liklihoods);
            classType = classes{maxClassLikIdx};%这里我是照着别人的python代码改的，还没完全看明白，但应该没啥问题
        end
    end
    
    methods(Access = private)
        %从一个特征集中读取参数并存储
        function classfeatureCol = getClassFeatureCol(~, dataSetFeatureCol, dataSetClasses, className)
            classfeatureCol = dataSetFeatureCol(ismember(dataSetClasses,className));
        end
        %计算均值、方差并村粗
        function structMeanVariance = calcMeanVariance(~, featureCol)
            mean = sum(featureCol)/numel(featureCol);
            variance = sum((featureCol-mean).^2)/(numel(featureCol)-1);
            structMeanVariance = struct('Mean', mean, 'Variance', variance);
        end
        %通过统计数据集类中出现的次数计算每个类的概率
        function [classes, classesProps] = calcClassProp(~, dataSetClasses)
            classes=unique(dataSetClasses,'stable');
            classesProps = cellfun(@(x) sum(ismember(dataSetClasses,x))/numel(dataSetClasses),classes,'un',0);
        end
        % 
        %通过一个特征计算其总似然
        function out = calcPartLiklihood(~, mean, variance, x)
            out = (exp(((x-mean)^2)/(2*variance))*sqrt(2*pi*variance))^-1;
        end
    end
end