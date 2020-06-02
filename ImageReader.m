classdef ImageReader
    %%%��Ҫ��ͼ�����ݶ�ȡ������ͱ��γ̹�ϵ���󣬶�����ʵ��������ǲο����ϵķ������������ѡ�񲻽�
    methods(Access = public)
        function [dataClasses, imagePaths2D] = read(~, DATASETPATH)
            %��dateset�ļ����ж�ȡȫ������
            DIRs = dir(DATASETPATH);
            % �������������ƣ�ÿ��������һ���ļ���������dataClasses��
            dataClasses = cell(numel(DIRs)-2,1);
            % ����ÿ���������ͼ��·�������鵥Ԫ����
            imagePaths2D = cell(size(dataClasses));
            parfor i=3:numel(DIRs) 
                %��ȡ����·��
                CLASSDIRPATH = strcat( DATASETPATH , '\' , DIRs(i).name );
                dataClasses{i-2} = strrep(DIRs(i).name,'_',''); %����������DIRs(i).name�е��»���
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
