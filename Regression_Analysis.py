import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import random
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn import svm


def X_percent(X,y, percent, test_size = .3):
	
	## define lists of INDICES to slice output arrays to ##
	idxs = np.random.choice(np.arange(len(y)), percent*len(y), replace = False)
	X_sample = X[idxs, :]
	y_sample = y[idxs]

	X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size = test_size)
		
	return (X_train, y_train, X_test, y_test)

def plot_learning_curve(estimator, X, y, iterations_per_percent = 10, title = None, ylim = None, train_sizes = np.linspace(.1,1.0,10)):
	
	plt.figure()
	if title is not None:
		plt.title(title)
	if ylim is not None:
		plt.ylim(ylim)
	plt.xlabel('Proportion of Training Data')
	plt.ylabel("Score")
	
	train_scores_mean = []
	test_scores_mean = []
	train_scores_std = []
	test_scores_std = []
	for size in train_sizes:

		train_scores = []
		test_scores = []
		
		for i in range(10): 

		# use only train_size percent of data. Convert X and y to numpy arrays.
			(X_train, y_train, X_test, y_test) = X_percent(X,y, percent = size)
		
		# fit and score #
			estimator.fit(X_train, y_train)
			y_test_pred = estimator.predict(X_test)
			(r_test,p) = pearsonr(y_test, y_test_pred)
			y_train_pred = estimator.predict(X_train)
			(r_train,p) = pearsonr(y_train, y_train_pred)

		
			# append to results.
			train_scores.append(r_train)
			test_scores.append(r_test)
		
		train_scores_mean.append(np.mean(train_scores))
		test_scores_mean.append(np.mean(test_scores))
		train_scores_std.append(np.std(train_scores))
		test_scores_std.append(np.std(test_scores))

	print("Starting Plot")
	
	plt.grid()
	#train_sizes = np.multiply(train_sizes, len(y))
	#plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	#				train_scores_mean + train_scores_std, alpha=0.1, color="r")
	#plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	#				test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			label="Testing score")

	plt.legend(loc="best")
	return estimator, plt, train_scores_mean, test_scores_mean

def randomized_list_parition(X, y, test_size = .3):
	p_size = int(np.floor(len(y)*.3))
	# starting at a random index, create p_size  consecutive indicies to slice X
	idx = random.randint(0, len(y))
	if idx + p_size > len(y):
		test_idxs1 = np.arange(idx, len(y), 1)
		test_idxs2 = np.arange(0, len(y)- idx, 1)
		test_idxs = np.concatenate((test_idxs1, test_idxs2))

		train_idxs = np.arange(len(y) - idx, idx, 1)


		y_test = np.take(y, test_idxs)
		y_train = np.take(y, train_idxs)
		

		X_test = np.take(X, test_idxs, axis = 0) 
		X_train = np.take(X, train_idxs, axis = 0)
	else: 
		test_idxs = np.arange(idx,idx+p_size, 1)
		train_idxs1 = np.arange(0,idx, 1)
		train_idxs2 = np.arange(idx+p_size, len(y), 1)
		train_idxs = np.concatenate((train_idxs1, train_idxs2))
		
		y_test = np.take(y, test_idxs)
		y_train = np.take(y, train_idxs)
		

		X_test = np.take(X, test_idxs, axis = 0) 
		X_train = np.take(X, train_idxs, axis = 0)

	return X_train, X_test, y_train, y_test


## import data as numeric type with cast ##
infile = "7_Day_Data.csv"
df = pd.read_csv(infile)
df = df.replace('-',0, regex = True)
df['H?'] = df['H?'].astype(float)
df['SB?'] = df['SB?'].astype(float)
df['CS?'] = df['CS?'].astype(float)
df['H?'] = df['AVG?'].astype(float)
df['H?'] = df['OBP?'].astype(float)
df['H?'] = df['SLG?'].astype(float)
df['TB?'] = df['TB?'].astype(float)
df['PA?'] = df['PA?'].astype(float)

## Removes pitchers from the dataset ##
df = df[df.Pos != 'P']

## Assign train and test features ##
y = df.Score.values

X = np.array(df)
X = np.delete(X,[0,1,2,3,4,5],1)

plt.boxplot(y)
plt.show()


pearsonr_test_scores = []
pearsonr_train_scores = []

for i in range(1):

	X_train, X_test, y_train, y_test = randomized_list_parition(X,y)
	
	
	#clf = Ridge(alpha =1)
	clf = MLPRegressor(hidden_layer_sizes= (20,5), activation = 'logistic', solver = 'adam')
	clf.fit(X_train, y_train)
	y_test_pred = clf.predict(X_test)
	(r_test,p) = pearsonr(y_test, y_test_pred)
	y_train_pred = clf.predict(X_train)
	(r_train,p) = pearsonr(y_train, y_train_pred)
	pearsonr_test_scores.append(r_test)
	pearsonr_train_scores.append(r_train)


	r_test_mean = np.mean(pearsonr_test_scores)
	r_test_std = np.std(pearsonr_test_scores)

clf, plt_lc, train_scores_mean, test_scores_mean = plot_learning_curve(clf, X, y, 
			title = '7 Day Training and Testing r values', train_sizes = np.linspace(.1,1,10))
plt_lc.savefig('Learning_Curve_Real_Data')
plt.show()
