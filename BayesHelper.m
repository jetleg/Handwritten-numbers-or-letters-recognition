classdef BayesHelper
    properties(Constant)    
    end  
    methods(Access = public)
        
        
        % �� baySet���Ϊmatrix(m,n)����mΪclass��������nΪ��������
        % ÿ�������ɾ�ֵ������ɵ�����Ԫ�ص�����
        function [baySet, classes, classesProps] = getBayesianSet(self, dataSet, dataSetClasses, normX)
            if nargin > 3
                dataSet = normX(dataSet); %��Ҫ�Ƚ����ݼ���׼��
            end
            [classes, classesProps] = self.calcClassProp(dataSetClasses);
            for classIdx=1:numel(classes)
                parfor featureIdx=1:numel(dataSet(1,:))
                    classfeatureCol = self.getClassFeatureCol(dataSet(:,featureIdx), dataSetClasses, classes{classIdx});
                    baySet(classIdx, featureIdx) = self.calcMeanVariance(classfeatureCol);
                end
            end
        end
        % ������ģ�͵ĸ��ʣ����������������Ӧ�����͡���������
        function [classType] = bayesClassify(self, baySet, classes, classesProps, newPattern)
            liklihoods = ones(size(classes));%likelihoodsƴ���ˣ������Ҳ���Ը�һ��
            parfor classIdx=1:numel(classes)
                for featureIdx=1:numel(baySet(1,:))
                    s = baySet(classIdx, featureIdx);
                    x = newPattern(featureIdx);
                    if liklihoods(classIdx) == 0
                        liklihoods(classIdx) = 0.0001;%�ȳ�ʼ��������
                    end
                    liklihoods(classIdx) = liklihoods(classIdx)*self.calcPartLiklihood(s.Mean, s.Variance, x);
                end
            end
            liklihoods = liklihoods.*cell2mat(classesProps);
            [~, maxClassLikIdx] = max(liklihoods);
            classType = classes{maxClassLikIdx};%�����������ű��˵�python����ĵģ���û��ȫ�����ף���Ӧ��ûɶ����
        end
    end
    
    methods(Access = private)
        %��һ���������ж�ȡ�������洢
        function classfeatureCol = getClassFeatureCol(~, dataSetFeatureCol, dataSetClasses, className)
            classfeatureCol = dataSetFeatureCol(ismember(dataSetClasses,className));
        end
        %�����ֵ��������
        function structMeanVariance = calcMeanVariance(~, featureCol)
            mean = sum(featureCol)/numel(featureCol);
            variance = sum((featureCol-mean).^2)/(numel(featureCol)-1);
            structMeanVariance = struct('Mean', mean, 'Variance', variance);
        end
        %ͨ��ͳ�����ݼ����г��ֵĴ�������ÿ����ĸ���
        function [classes, classesProps] = calcClassProp(~, dataSetClasses)
            classes=unique(dataSetClasses,'stable');
            classesProps = cellfun(@(x) sum(ismember(dataSetClasses,x))/numel(dataSetClasses),classes,'un',0);
        end
        % 
        %ͨ��һ����������������Ȼ
        function out = calcPartLiklihood(~, mean, variance, x)
            out = (exp(((x-mean)^2)/(2*variance))*sqrt(2*pi*variance))^-1;
        end
    end
end