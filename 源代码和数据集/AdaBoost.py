import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# 读入数据
default = pd.read_excel('../dataset/default of credit card.xls')
# 为确保绘制的饼图为圆形，需执行如下代码
plt.axes(aspect = 'equal')
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 统计客户是否违约的频数
counts = default.y.value_counts()
# 绘制饼图
plt.pie(x = counts, # 绘图数据
        labels = pd.Series(counts.index).map({0:'不违约',1:'违约'}),  # 添加文字标签
        autopct = '%.lf%%'  # 设置百分比的格式，这里保留一位小数
        )
# 显示图形
plt.show()
   

# 排除数据集中的ID变量和因变量，剩余的数据用作自变量x
X = default.drop(['ID','y'], axis = 1)
y = default.y
# 数据拆分
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25,random_state = 1234)
# 构建AdaBoost算法的类
AdaBoost1 = ensemble.AdaBoostClassifier()
# 算法在训练集上的拟合
AdaBoost1.fit(X_train, y_train)
# 算法在测试集上的预测
pred1 = AdaBoost1.predict(X_test)
# 返回模型的预测结果
print('模型的准确率为：\n',metrics.accuracy_score(y_test, pred1))
print('模型的评估报告：\n',metrics.classification_report(y_test, pred1))


# 计算客户违约的概率值，用于生成ROC曲线的数据
y_score = AdaBoost1.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr, tpr)
# 绘制面积图
plt.stackplot(fpr, tpr, color = 'steelblue', alpha = 0.5, edgecolor = 'black')
# 添加边际线
plt.plot(fpr, tpr, color = 'black', lw = 1)
# 添加对角线
plt.plot([0,1], [0,1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' %roc_auc)
# 添加x轴与y轴标签
plt.xlabel('l-specificity')
plt.ylabel('sensitivity')
# 显示图形
plt.show()


# 自变量的重要性排序
importance = pd.Series(AdaBoost1.feature_importances_, index = X.columns)
importance.sort_values().plot(kind = 'barh')
plt.show()


# 取出重要性比较高的自变量建模
predictors = list(importance[importance > 0.02].index)
# 通过网格搜索法选择基础模型所对应的合理参数组合
max_depth = [3, 4, 5, 6]
params1 = {'base_estimator__max_depth':max_depth}
base_model = GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator = 
                        DecisionTreeClassifier()),param_grid = params1,
                        scoring = 'roc_auc', cv = 5, n_jobs =4, verbose = 1)
base_model.fit(X_train[predictors], y_train)
n_estimators = [100, 200, 300]
learning_rate = [0.01, 0.05, 0.1, 0.2]
params2 = {'n_estimators':n_estimators, 'learning_rate':learning_rate}
adaboost = GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator =
          DecisionTreeClassifier(max_depth = 3)),param_grid = params2,
          scoring = 'roc_auc', cv = 5, n_jobs = 4, verbose = 1)
adaboost.fit(X_train[predictors], y_train)


# 使用最佳的参数组合构建AdaBoost模型
AdaBoost2 = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3),
            n_estimators = 300, learning_rate = 0.01)
# 算法在训练数据集上的拟合
AdaBoost2.fit(X_train[predictors], y_train)
# 算法在测试数据集上的预测
pred2 = AdaBoost2.predict(X_test[predictors])
# 返回模型的预测结果
print('模型的准确率为：\n', metrics.accuracy_score(y_test, pred2))
print('模型的评估报告：\n', metrics.classification_report(y_test, pred2))






