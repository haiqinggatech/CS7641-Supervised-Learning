import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from random import sample
from collections import defaultdict
import graphviz 
import pandas as pd
from sklearn.model_selection import GridSearchCV, learning_curve
from datetime import datetime
from sklearn.metrics import confusion_matrix

data = pd.read_excel(open('default of credit card clients.xls','rb'), sheetname='Data', skiprows=1)
data.head()
data.shape
data.columns
data.isnull().values.any()
data[data.isnull().any(axis=1)] 
data ['a'] = pd.DataFrame({'a':range(30001)})
sampled_df = data[(data['a'] % 10) == 0]
sampled_df.shape
sampled_df_remaining = data[(data['a'] % 10) != 0]
sampled_df_remaining.shape
LoanY = sampled_df['default payment next month'].copy()
loan_features = ['LIMIT_BAL','SEX','EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
LoanX = sampled_df[loan_features].copy()
classifed_names= ['default payment next month']

trainTest = train_test_split(LoanX, LoanY, test_size=0.3, train_size=0.7, random_state=0)



from sklearn.model_selection import GridSearchCV, learning_curve
from datetime import datetime
from sklearn.metrics import confusion_matrix

cross_validations = 10
train_sizes_base = [100, 200, 400, 600,800,1000,1200,1400]

def plot_learning_curve(title, cv_curve):
#     _, _, test_scores_base = base_curve
    train_sizes, train_scores, test_scores = cv_curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
#     test_scores_base_mean = np.mean(test_scores_base, axis=1)
#     test_scores_base_std = np.std(test_scores_base, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.ylim((.6, 1.01))
    
    plt.ylim((.6, 1.01))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    
#     plt.fill_between(train_sizes, test_scores_base_mean - test_scores_base_std,
#                      test_scores_base_mean + test_scores_base_std, alpha=0.1, color="b")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
#     plt.plot(train_sizes, test_scores_base_mean, 'o-', color="b",
#              label="Test Score without CV")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test Score with CV")

    plt.legend(loc="best")
    return plt

def crossValidateAndTest(name, clf, params, trainTest, scaler=None, plot=True):
    X_train, X_test, y_train, y_test = trainTest.copy()
    print('Name: ' + name)
    if not scaler is None:
        scaler.fit(X_train) 
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
#     base_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=None)
    cv_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=cross_validations)
    
#     base_clf.fit(X_train, y_train)
    
    start = datetime.now()
    cv_clf.fit(X_train, y_train)
    end = datetime.now()
    train_time = (end - start).total_seconds()
    train_score = cv_clf.score(X_train, y_train)
    print('Train time: ' + str(train_time))
    print('Train score: ' + str(train_score))
    
    start = datetime.now()
    test_score = cv_clf.score(X_test, y_test)
    end = datetime.now()
    test_time = (end - start).total_seconds()
    print('Test time: ' + str(test_time))
    print('Test score: ' + str(test_score))
    
    y_predict = cv_clf.predict(X_test)
    confusion_results = confusion_matrix(y_test, y_predict)
    print(confusion_results)
    
    
#     base_estimator = base_clf.best_estimator_
    cv_estimator = cv_clf.best_estimator_
    
    print("Best params: " + str(cv_clf.best_params_))
    
    table_output = '{:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(train_score, train_time, test_score, test_time)
    
    
    
#     score = optimized_clf.score(X_test, y_test)
        
    all_sizes = train_sizes_base 
#     base_curve = learning_curve(base_estimator, X_train, y_train, cv=None, train_sizes=all_sizes)

    if plot:
        cv_curve = learning_curve(cv_estimator, X_train, y_train, cv=cross_validations, train_sizes=all_sizes)

    #     plot = plot_learning_curve(name, base_curve, cv_curve)
        plot = plot_learning_curve(name, cv_curve) 

    
    return (cv_estimator, cv_clf, table_output)

print("Finished")


# In[ ]:


# Decision Tree
params = { 'criterion':['gini','entropy'] }

for i in [1, 3, 6, 10, 15, 20, 25, 35, 50]: 
    clf = tree.DecisionTreeClassifier(max_depth=i, class_weight='balanced', splitter='best', min_samples_leaf=1)
    output, output_clf, table_output = crossValidateAndTest('Decision Tree: ' + str(i), clf, params, trainTest)
    tree_size = output.tree_.node_count
    print('{} & {} & {} & '.format(i, output_clf.best_params_['criterion'], tree_size) + table_output + ' \\\\ \\hline') 
    
    print()
#     clf = clf.fit(X_train, y_train)

#     y_predict = clf.predict(X_test)
#     scores = cross_val_score(clf, X, y)
#     print(str(i) + ': ' + str(accuracy_score(y_test, y_predict)) + '   |   ' + str(scores.mean()))
    if i == 7:
        dot_data = tree.export_graphviz(output, out_file=None, 
                             feature_names=df_x,  
                             class_names=classifed_names,  
                             filled=True, rounded=True,  
                             special_characters=True)  
        graph = graphviz.Source(dot_data).view()


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
estimators = [1, 3, 5, 15, 50, 100, 150]
learning_rate = [.1, 1, 10]

for i in [1, 3, 5, 10, 15, 20]: 
    clf_base = tree.DecisionTreeClassifier(max_depth=i, criterion='gini', splitter='best')
    clf = AdaBoostClassifier(base_estimator=clf_base)
    output, output_clf, table_output = crossValidateAndTest('Adaboost: ' + str(i), clf, {'n_estimators': estimators, 'learning_rate': learning_rate}, trainTest)
    tree_count = len(output)
    print('{} & {} & {} & '.format(i, output_clf.best_params_['learning_rate'], output_clf.best_params_['n_estimators']) + table_output + ' \\\\ \\hline') 


# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  


alphas = [0.0001, ]#0.005, 0.001]
learning_rate_inits = [ 0.001, ]#.01, .1, .2]
activations = ['relu']
layers = []

for i in [5,10,100]:
    for j in [1, 3]:
        layers.append((i, j))
        
# params = { 'alpha': alphas, 'learning_rate_init': learning_rate_init}
params = {}

print("Starting neural networks")
for alpha in alphas:
    for learning_rate_init in learning_rate_inits:
        clf = MLPClassifier(solver='adam', max_iter=2000, random_state=7, batch_size='auto')
        plot = True
        output, output_clf, table_output = crossValidateAndTest('Adam NN: ' + str(alpha) + '/' + str(learning_rate_init), clf, params, trainTest, StandardScaler(), plot)
        print(str(output.n_iter_))
        print('{} & {} & '.format(alpha, learning_rate_init) + table_output + ' \\\\ \\hline') 
        
        clf = MLPClassifier(solver='sgd', max_iter=2000, random_state=7, batch_size='auto')
        plot = True
        output, output_clf, table_output = crossValidateAndTest('SGD NN: ' + str(alpha) + '/' + str(learning_rate_init), clf, params, trainTest, StandardScaler(), plot)
        print(str(output.n_iter_))
        print('{} & {} & '.format(alpha, learning_rate_init) + table_output + ' \\\\ \\hline') 





# In[ ]:


import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
#     ax.xlabel("# Iterations")
#     ax.ylabel("Loss")
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 1000

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        
        start = datetime.now()
        mlp.fit(X, y)
        end = datetime.now()
        train_time = (end - start).total_seconds()
        
        start = datetime.now()
        score = mlp.score(X, y)
        end = datetime.now()
        score_time = (end - start).total_seconds()
        
        table_output = '{} & {:.4f} & {:.4f} & {:.4f}'.format(label, score, mlp.loss_, train_time)
        print(table_output)

        
        mlps.append(mlp)
        print("Training set score: %f" % score)
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)
        


fig, axes = plt.subplots(1, 1, figsize=(15, 5))
# load / generate some toy datasets

ax = axes
plot_on_dataset(X, y, ax, 'Test')

plt.xlabel("# Iterations")
plt.ylabel("Loss")
fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()


# In[ ]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler  

kernels = ['rbf', 'poly']
gammas = [0.01, .05, 1.0, 2.0]
for kernel in kernels:
    for gamma in gammas:
        clf_1 = svm.SVC(kernel=kernel, max_iter=30000, gamma=gamma)
        output, output_clf, table_output = crossValidateAndTest('SVM ' + kernel + ' - ' + str(gamma), clf_1, {}, trainTest, StandardScaler())
        print('{} & {} & '.format(kernel, gamma) + table_output + ' \\\\ \\hline') 


# In[ ]:


from sklearn import neighbors

print('Running knn')
weights = ['uniform', 'distance']
k_vals = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]

outputs = []
for weight in weights:
    for k in k_vals:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(k, weights=weight)
        output, output_clf, table_output = crossValidateAndTest('KNN - ' + weight + ': ' + str(k), clf, {}, trainTest)
        print('{} & {} & '.format(weight, k) + table_output + ' \\\\ \\hline') 
        outputs.append('{} & {} & '.format(weight, k) + table_output + ' \\\\ \\hline')
        print()


for line in outputs:
    print(line)


