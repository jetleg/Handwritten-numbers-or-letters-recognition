classdef Classifier
    
    properties(Constant)
        tr = Trainer;
        bh = BayesHelper;
    end
    properties(Access = private)
        svm = SVMHelper;
    end
    
    methods(Access = public)

        function [testObjects, testObjectsPositions] = getImgReady(self, ImgPath, noiseThreshold, blockSize)
            switch nargin
                case 3
                    blockSize = self.tr.defBlockSize;
                case 2
                    blockSize = self.tr.defBlockSize;
                    noiseThreshold = self.tr.defNoiseThreshold;
            end
            try
                [testObjects, ~, testObjectsPositions] = self.tr.Train({'Unknown'}, {{ImgPath}}, noiseThreshold, blockSize);
            catch
            end
        end
        
        function [testObjects, testObjectsPositions] = getImgReadyHOG(self, ImgPath, noiseThreshold, CellSize)
            switch nargin
                case 3
                    CellSize = self.tr.defBlockSize;
                case 2
                    CellSize = self.tr.defBlockSize;
                    noiseThreshold = self.tr.defNoiseThreshold;
            end
            try
                [testObjects, ~, testObjectsPositions] = self.tr.TrainHOG({'Unknown'}, {{ImgPath}}, noiseThreshold, CellSize);
            catch
            end
        end
        
      
        
                
        function [classesTypes] = bayesClassifyAsync(self, baySet, classes, classesProps, testPatterns, normX)
            if nargin > 5
                testPatterns = normX(testPatterns); 
            end
            parfor objIdx=1:numel(testPatterns(:,1))
                classesTypes{objIdx,1} = self.bh.bayesClassify(baySet, classes, classesProps, testPatterns(objIdx,:));
            end
        end
            
    end 
end
