classdef ImageReader
    %%%主要是图像数据读取，这里和本课程关系不大，而且其实大多数都是参考网上的方法，这里可以选择不讲
    methods(Access = public)
        function [dataClasses, imagePaths2D] = read(~, DATASETPATH)
            %从dateset文件夹中读取全部数据
            DIRs = dir(DATASETPATH);
            % 保存所有类名称（每个名称是一个文件夹名）于dataClasses中
            dataClasses = cell(numel(DIRs)-2,1);
            % 保存每个类的所有图像路径的数组单元数组
            imagePaths2D = cell(size(dataClasses));
            parfor i=3:numel(DIRs) 
                %获取数据路径
                CLASSDIRPATH = strcat( DATASETPATH , '\' , DIRs(i).name );
                dataClasses{i-2} = strrep(DIRs(i).name,'_',''); %这里是消除DIRs(i).name中的下划线
                Files = dir(CLASSDIRPATH);
                    imagePaths = cell(numel(Files)-2,1);
                    for j=3:numel(Files)
                        imagePaths{j-2} = strcat( CLASSDIRPATH, '\', Files(j).name);
                    end
                    imagePaths2D{i-2} = imagePaths;
            end
        end
    end
end
