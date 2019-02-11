import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from random import sample
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import GridSearchCV, learning_curve
from datetime import datetime
from sklearn.metrics import confusion_matrix
import re

adult = pd.read_hdf('datasets.hdf','adult')  
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values



df_x=['age', 'edu_num', 'hrs',  'cap_gain_loss', 'employer_?', 'employer_Federal_gov', 'employer_Local_gov', 'employer_Never_worked', 'employer_Private', 'employer_Self_emp_inc', 'employer_Self_emp_not_inc', 'employer_State_gov', 'marital_Divorced', 'marital_Married_AF_spouse', 'marital_Married_civ_spouse', 'marital_Married_spouse_absent', 'marital_Never_married', 'marital_Separated', 'marital_Widowed', 'occupation_?', 'occupation_Adm_clerical', 'occupation_Armed_Forces', 'occupation_Craft_repair', 'occupation_Exec_managerial', 'occupation_Farming_fishing', 'occupation_Handlers_cleaners', 'occupation_Machine_op_inspct', 'occupation_Other_service', 'occupation_Priv_house_serv', 'occupation_Prof_specialty', 'occupation_Protective_serv', 'occupation_Sales', 'occupation_Tech_support', 'occupation_Transport_moving', 'relationship_Not_in_family', 'relationship_Other_relative', 'relationship_Own_child', 'relationship_Spouse', 'relationship_Unmarried', 'race_Amer_Indian_Eskimo', 'race_Asian_Pac_Islander', 'race_Black', 'race_Other', 'race_White', 'sex_Female', 'sex_Male', 'country_?', 'country_British_Commonwealth', 'country_China', 'country_Euro_1', 'country_Euro_2', 'country_Latin_America', 'country_Other', 'country_SE_Asia', 'country_South_America', 'country_United_States']

classifed_names = ['income']
trainTest = train_test_split(adultX, adultY, test_size=0.3, train_size=0.7, random_state=0)


from sklearn.model_selection import GridSearchCV, learning_curve
from datetime import datetime
from sklearn.metrics import confusion_matrix

cross_validations = 10
train_sizes_base = [500, 1000, 2500, 5000, 10000, 15000]




def plot_learning_curve(title, cv_curve):
#     _, _, test_scores_base = base_curve
    train_sizes, train_scores, test_scores = cv_curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.ylim((.6, 1.01))
    

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
#     plt.plot(train_sizes, test_scores_base_mean, 'o-', color="b",
#              label="Test Score without CV")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test Score with CV")

    plt.legend(loc="best")
    plt.savefig(title)
    plt.close()
    return plt



def crossValidateAndTest(name, clf, params, trainTest, scaler=None):
    X_train, X_test, y_train, y_test = trainTest.copy()
    print('Name: ' + name)
    if not scaler is None:
        scaler.fit(X_train) 
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
#     base_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=None)
    cv_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=cross_validations)
    
    
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
    
    table_output = 'train_score {:.4f} & train_time {:.4f} & test_score{:.4f} & test_time {:.4f}'.format(train_score, train_time, test_score, test_time)
    
    
        
    all_sizes = train_sizes_base 

    cv_curve = learning_curve(cv_estimator, X_train, y_train, cv=cross_validations, train_sizes=all_sizes)


    plot = plot_learning_curve(name, cv_curve) 

    
    return (cv_estimator, cv_clf, table_output)

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
    for i in range(len(learning_rate_inits)):
        clf = MLPClassifier(solver='adam', max_iter=2000, random_state=7, batch_size='auto')
        plot = True
        output, output_clf, table_output = crossValidateAndTest('Adam NN: ' + str(alpha), clf, params, trainTest, StandardScaler())
        print(str(output.n_iter_))
        print('{} & {} & '.format(alpha, learning_rate_inits[i]) + table_output + ' \\\\ \\hline') 
        
        clf = MLPClassifier(solver='sgd', max_iter=2000, random_state=7, batch_size='auto')
        plot = True
        output, output_clf, table_output = crossValidateAndTest('SGD NN: ' + str(alpha), clf, params, trainTest, StandardScaler())
        print(str(output.n_iter_))
        print('{} & {} & '.format(alpha, learning_rate_inits[i]) + table_output + ' \\\\ \\hline') 





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
