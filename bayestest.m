%一个简单的测试，大概测试一下分类效果
%dataset中每组数据有三种特征，分别代表鞋码（欧洲标准）、身高（cm)、体重(kg)，分类结果F表示女性，M表示男性
%test.m只是为了得到一个大概的效果估计，dataset中的数据是并不是实际获得的，是根据经验填写的
dataclasses = {'M','M','M','M', 'F', 'F', 'F', 'F'};%用四男四女
dataset = [42 180 80;43 190 100; 40 170 75; 37 165 50; 35 150 40 ; 35 155 60; 36 160 50; 36 160 50 ];
testman = [35 150 80];%这里你现场演示的时候可以改一改给看看效果，预测结果大致还是符合人们的预期的
%其实还是有点不严谨的，比如输入35 150 80他会输出F，但实际上应该更可能是一个比较胖的女孩
%因为特征太少了，可以加入比如年龄、是否爱好打游戏、是否爱喝奶茶等等比较有区分度的特征
%当然这里只是一个简单的测试，不过多费周章
cl = Classifier;
[baySet, classes, classesProps] = cl.bh.getBayesianSet(dataset, dataclasses);
classType = cl.bh.bayesClassify(baySet, classes, classesProps, testman);
disp(classType);
