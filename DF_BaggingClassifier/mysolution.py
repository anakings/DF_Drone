from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from get_csv_data import HandleData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def MethodGSCV(estimator, tuned_parameters):
	scores = ['precision']

	for score in scores:
		#print("# Tuning hyper-parameters for %s" % score)

	    clf = GridSearchCV(estimator, tuned_parameters, scoring='%s_macro' % score)
	    #print("Best parameters set found on development set:\n")
	    clf.fit(X, y)
	    print(clf.best_params_)

	    print("Grid scores on development set:")
	    print()
	    means = clf.cv_results_['mean_test_score']
	    stds = clf.cv_results_['std_test_score']
	    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

def scores(train_scores,test_scores):
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std

def plot_validation_curve(estimator, title, X, y, param_name, param_range, xlabel=None, ylim=None, n_jobs=None):
	
	train_scores, test_scores = validation_curve(
    estimator, X, y, param_name=param_name, param_range=param_range,
    scoring="accuracy", n_jobs=n_jobs)
	
	train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	if xlabel is None:
		xlabel = param_name
	plt.xlabel(xlabel)
	plt.ylabel("Score")
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
    	         color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    	         color="navy", lw=lw)
	plt.legend(loc="best")
	plt.savefig('./images resul/' + title + '.png', dpi=600)
	plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), savefig=None):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
                       # , return_times=True)
    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

    for index, value in enumerate(train_sizes):
    	#print(index)
    	print('train_scores:', train_scores_mean[index], 'test_scores', test_scores_mean[index], 'for {', value, '}')
    
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    if savefig == None:
    	savefig = title
    plt.savefig('./images resul/' + savefig + '.png', dpi=600)
    plt.show()

if __name__ == '__main__':    
    
	#get the data
	data = HandleData(oneHotFlag=False)
	X, y = data.get_synthatic_data()
	print('data_sizes:', X.shape)
	print('label_sizes:', len(y))

	######################################   SVC   ###############################################################
	estimator = SVC()

	# Set the parameters by cross-validation
	tuned_parameters = [{'kernel': ['rbf','poly','sigmoid'], 'gamma': np.logspace(-6, 10, 5), 'C': [1, 10, 100, 1000,10000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]}]

	x = MethodGSCV(estimator, tuned_parameters)

	#plot Validation Curve
	title = "Validation Curve (SVM)_C"
	plot_validation_curve(estimator, title, X, y, ylim=(0.7, 1.01), param_name = 'C', param_range = [1, 10, 100, 1000, 10000])
	title = "Validation Curve (SVM)_gamma"
	plot_validation_curve(estimator, title, X, y, ylim=(0.7, 1.01), param_name = 'gamma', param_range = np.logspace(-1, 10, 5), xlabel = r"$\gamma$")

	#plot Learning Curve
	title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
	cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	estimator = SVC(C=10,gamma=100) #rbf kernel is default
	plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 10), savefig="Learning Curves (SVM)")


	######################################   DecisionTreeClassifier   ###############################################################
	estimator = DecisionTreeClassifier()

	# Set the parameters by cross-validation
	tuned_parameters = [{'max_depth': [x for x in range(1,25)], 'min_samples_split': np.linspace(0.1, 1.0, 10)},
	{'max_depth': [x for x in range(1,25)]}]

	x = MethodGSCV(estimator, tuned_parameters)

	#plot Validation Curve
	title = "Validation Curve (DT)_max_depth"
	plot_validation_curve(estimator, title, X, y, ylim=(0.2, 1.01), param_name = 'max_depth', param_range = [x for x in range(1,25)])
	title = "Validation Curve (DT)_min_samples_split"
	plot_validation_curve(estimator, title, X, y, ylim=(0.5, 1.01), param_name = 'min_samples_split', param_range = np.linspace(0.1, 1.0, 10))


	#plot Learning Curve
	title = r"Learning Curves (DT)"
	cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	estimator = DecisionTreeClassifier(max_depth=16)
	plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 10))
	

	######################################   BaggingClassifier   ###############################################################
	######################################   DecisionTreeClassifier   ###############################################################
	
	estimator = BaggingClassifier(DecisionTreeClassifier(max_depth=16))

	# Set the parameters by cross-validation
	tuned_parameters = [{'n_estimators': [x for x in range(100,1100,100)], 'max_samples': np.linspace(0.1, 1.0, 5)}]

	x = MethodGSCV(estimator, tuned_parameters)

	#plot Validation Curve
	title = "Validation Curve (BC_DT)_n_estimators"
	plot_validation_curve(estimator, title, X, y, ylim=(0.2, 1.01), param_name = 'n_estimators', param_range = [x for x in range(100,1100,100)])
	title = "Validation Curve (BC_DT)_max_samples"
	plot_validation_curve(estimator, title, X, y, ylim=(0.5, 1.01), param_name = 'max_samples', param_range = np.linspace(0.1, 1.0, 5))


	#plot Learning Curve
	title = r"Learning Curves (BC_DT)"
	cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	BaggingClassifier(DecisionTreeClassifier(random_state=42, max_depth=16), n_estimators=300, max_samples=0.55, bootstrap=False, n_jobs=-1, random_state=42)
	plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 10))

	######################################################################################################################################
	modelList = [SVC(C=10,gamma=100), DecisionTreeClassifier(max_depth=16), BaggingClassifier(DecisionTreeClassifier(max_depth=16), n_estimators=300, max_samples=0.55, bootstrap=False, n_jobs=-1, random_state=42)]
	modelName = ['SVC', 'Decision Tree Classifier', 'Bagging Classifier']

	#modelList = [DecisionTreeClassifier(max_depth=16), BaggingClassifier(DecisionTreeClassifier(max_depth=16), n_estimators=300, max_samples=0.55, bootstrap=False, n_jobs=-1, random_state=42)]
	#modelName = ['Decision Tree Classifier','Bagging Classifier']
	test_percentage = [0.1,0.3,0.3]
	
	for name,model in enumerate(modelList):
		antenna_data, antenna_data_test, label_data, label_test = train_test_split (X, y, test_size=test_percentage[name], random_state=42)
		
		clf = model
		tic = time()
		clf.fit(antenna_data, label_data)
		#Predict
		y_pred = clf.predict(antenna_data_test)
			
		print('Accuracy validation of model', modelName[name], 'is:', accuracy_score(label_test, y_pred)*100, '%')
		print('Elapsed time:', time() - tic, 'seconds')
		
		mat = confusion_matrix(label_test, y_pred)
		mat = np.round(mat / mat.astype(np.float).sum(axis=0) *100)
		mat = mat.astype(int)
		fig = plt.figure(figsize=(6, 6))
		sns.set()
		sns.heatmap(mat.T, square=True, annot=True, fmt='', cbar=False, xticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], \
			yticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], cmap="Blues")
		plt.xlabel('true label')
		plt.ylabel('predicted label')
		plt.savefig('./images resul/confusion_matrix_' + 'modelName[name]' + str(int(test_percentage[name]*100)) + '%.png', dpi=600)
		plt.show()
		

