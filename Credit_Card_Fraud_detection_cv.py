#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imported Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm
from sklearn.neural_network import BernoulliRBM

# Other Libraries
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.externals import joblib
import tensorflow as tf
from datetime import datetime 
from sklearn.metrics import recall_score 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score

# path = '/Users/Jill/Documents/Internship_Project/fraud_transaction_detection/'
df = pd.read_csv('creditcard.csv')
df.head()
df.describe()
# Good No Null Values!
df.isnull().sum().max()

df.columns
# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


#plt.show()

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

df.head()

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X_org = df.drop('Class', axis=1)
y_org = df['Class']

predictions_log_reg = np.zeros(len(X_org),np.float32)
labels_log_reg = np.zeros(len(y_org),np.uint8)

predictions_knn = np.zeros(len(X_org),np.float32)
labels_knn = np.zeros(len(y_org),np.uint8)

predictions_tree = np.zeros(len(X_org),np.float32)
labels_tree = np.zeros(len(y_org),np.uint8)

predictions_forest = np.zeros(len(X_org),np.float32)
labels_forest = np.zeros(len(y_org),np.uint8)

predictions_xgboost = np.zeros(len(X_org),np.float32)
labels_xgboost = np.zeros(len(y_org),np.uint8)

predictions_svm = np.zeros(len(X_org),np.float32)
labels_svm = np.zeros(len(y_org),np.uint8)

#do 5-fold cross validation
folds = 5
skf = StratifiedKFold(n_splits=folds, random_state=42, shuffle=False)
fold_num = 0
for train_index, test_index in skf.split(X_org, y_org):
	print('processing fold {:d} over {:}'.format(fold_num+1, folds))
	fold_num = fold_num + 1
	original_Xtrain, original_Xtest = X_org.iloc[train_index], X_org.iloc[test_index]
	original_ytrain, original_ytest = y_org.iloc[train_index], y_org.iloc[test_index]

	# Turn into an array

	original_Xtrain = original_Xtrain.values
	original_Xtest = original_Xtest.values
	original_ytrain = original_ytrain.values
	original_ytest = original_ytest.values

	# See if both the train and test label distribution are similarly distributed
	train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
	test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
	print('-' * 100)

	print('Label Distributions: \n')
	print(train_counts_label/ len(original_ytrain))
	print(test_counts_label/ len(original_ytest))

	# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

	# Lets shuffle the data before creating the subsamples

	df = df.sample(frac=1) #shuffle the data by rows

	# amount of fraud classes 492 rows.
	fraud_df = df.loc[df['Class'] == 1]
	non_fraud_df = df.loc[df['Class'] == 0][:len(fraud_df)]

	normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

	# Shuffle dataframe rows
	new_df = normal_distributed_df.sample(frac=1, random_state=42)

	new_df.head()

	print('Distribution of the Classes in the subsample dataset')
	print(new_df['Class'].value_counts()/len(new_df))

	# Undersampling before cross validating (prone to overfit)
	X = new_df.drop('Class', axis=1)
	y = new_df['Class']

	# Our data is already scaled we should split our training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Turn the values into an array for feeding the classification algorithms.
	X_train = X_train.values
	X_test = X_test.values
	y_train = y_train.values
	y_test = y_test.values

	# Supervisied Models
	# Classifier for random undersampling

	classifiers = {
	    "LogisiticRegression": LogisticRegression(),
	    "KNearest": KNeighborsClassifier(),
	    "Support Vector Classifier": SVC(),
	    "DecisionTreeClassifier": DecisionTreeClassifier(),
	    "RandomForestClassifier": RandomForestClassifier(),
	    "XgboostClaasifier":  xgb.XGBClassifier()
	}


	for key, classifier in classifiers.items():
	    classifier.fit(X_train, y_train)
	    training_score = cross_val_score(classifier, X_train, y_train, cv=5,  scoring='recall')
	    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% recall score")

	# Use GridSearchCV to find the best parameters.
	# Logistic Regression 
	log_reg_params = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
	grid_log_reg.fit(X_train, y_train)
	# We automatically get the logistic regression with the best parameters.
	log_reg = grid_log_reg.best_estimator_
	print("best parameters of Logistic Regression: ", grid_log_reg.best_params_)
	joblib.dump(log_reg, "log_reg.m")

	#K-Nearest-Neighbor
	knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
	grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
	grid_knears.fit(X_train, y_train)
	# KNears best estimator
	knears_neighbors = grid_knears.best_estimator_
	print("best parameters of K-nearest Neighbors: ", grid_knears.best_params_)
	joblib.dump(knears_neighbors, "knears_neighbors.m")

	# Support Vector Classifier
	svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'probability': [True]}
	grid_svc = GridSearchCV(SVC(), svc_params)
	grid_svc.fit(X_train, y_train)
	# SVC best estimator
	svc = grid_svc.best_estimator_
	print("best parameters of SVM: ", grid_svc.best_params_)
	joblib.dump(svc, "svc.m")

	# DecisionTree Classifier
	tree_params = {"criterion": ['gini', 'entropy'], 'max_depth': list(range(2,4,1)), 
	              "min_samples_leaf": list(range(5,7,1))}
	grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
	grid_tree.fit(X_train, y_train)
	# tree best estimator
	tree_clf = grid_tree.best_estimator_
	print("best parameters of Decision Tree: ", grid_tree.best_params_)
	joblib.dump(tree_clf, "tree_clf.m")

	#RandomForest Classifier
	forest_params = {'n_estimators': [10, 15, 20, 30], 'oob_score': ['True']}
	grid_forest = GridSearchCV(RandomForestClassifier(), forest_params)
	grid_forest.fit(X_train, y_train)
	#RandomForest best estimator
	forest = grid_forest.best_estimator_
	print("best parameters of Random Forest: ", grid_forest.best_params_)
	joblib.dump(forest, "forest.m")

	#Xgboost Classifier
	xgboost_params = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5], 'max_depth': [4, 5, 6]}
	grid_xgboost = GridSearchCV(xgb.XGBClassifier(), xgboost_params)
	grid_xgboost.fit(X_train, y_train)
	#Xgboost best estimator
	xgboost = grid_xgboost.best_estimator_
	print("best parameters of Xgboost: ", grid_xgboost.best_params_)
	joblib.dump(xgboost, "xgboost.m")

#for probability
	log_reg = joblib.load("log_reg.m")
	log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5)
    
	knears_neighbors = joblib.load("knears_neighbors.m")
	knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
    
	svc = joblib.load("svc.m")
	svc_pred = cross_val_predict(svc, X_train, y_train, cv=5)
    
	tree_clf = joblib.load("tree_clf.m")
	tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
    
	forest = joblib.load("forest.m")
	forest_pred = cross_val_predict(forest, X_train, y_train, cv=5)
    
	xgboost = joblib.load("xgboost.m")
	xgboost_pred = cross_val_predict(xgboost, X_train, y_train, cv=5)
    
#Test
	log_reg_test = log_reg.predict_proba(original_Xtest)
	log_reg_test = log_reg_test[:, 1]
	predictions_log_reg[test_index] = log_reg_test
	labels_log_reg[test_index] = original_ytest

	knears_test = knears_neighbors.predict_proba(original_Xtest)
	knears_test = knears_test[:, 1]
	predictions_knn[test_index] = knears_test
	labels_knn[test_index] = original_ytest
    
	svc_test = svc.predict_proba(original_Xtest)
	svc_test = svc_test[:, 1]
	#svc_test = svc.predict(original_Xtest)
	predictions_svm[test_index] = svc_test
	labels_svm [test_index] = original_ytest
    
	tree_clf_test = tree_clf.predict_proba(original_Xtest)
	tree_clf_test = tree_clf_test[:, 1]
	predictions_tree[test_index] = tree_clf_test
	labels_tree[test_index] = original_ytest
    
	forest_test = forest.predict_proba(original_Xtest)
	forest_test = forest_test[:, 1]
	predictions_forest[test_index] = forest_test
	labels_forest[test_index] = original_ytest
    
	xgboost_test = xgboost.predict_proba(original_Xtest)
	xgboost_test = xgboost_test[:, 1]
	predictions_xgboost[test_index] = xgboost_test
	labels_xgboost[test_index] = original_ytest
    
#ROC Curve of Supervised Models
log_fpr, log_tpr, log_thresold = roc_curve(labels_log_reg, predictions_log_reg)
log_reg_auc = auc(log_tpr, log_fpr)

knear_fpr, knear_tpr, knear_threshold = roc_curve(labels_knn, predictions_knn)
knear_auc = auc(knear_tpr, knear_fpr)

svc_fpr, svc_tpr, svc_threshold = roc_curve(labels_svm, predictions_svm)
svc_auc = auc(svc_tpr, svc_fpr)

tree_fpr, tree_tpr, tree_threshold = roc_curve(labels_tree, predictions_tree)
tree_auc = auc(tree_tpr, tree_fpr)

forest_fpr, forest_tpr, forest_threshold = roc_curve(labels_forest,predictions_forest)
forest_auc = auc(forest_tpr, forest_fpr)

xgboost_fpr, xgboost_tpr, xgboost_threshold = roc_curve(labels_xgboost, predictions_xgboost)
xgboost_auc = auc(xgboost_tpr, xgboost_fpr)

#ROC Curve of Supervised Models
log_fpr, log_tpr, log_thresold = roc_curve(original_ytest, log_reg_test)
knear_fpr, knear_tpr, knear_threshold = roc_curve(original_ytest, knears_test)
svc_fpr, svc_tpr, svc_threshold = roc_curve(original_ytest, svc_test)
tree_fpr, tree_tpr, tree_threshold = roc_curve(original_ytest, tree_clf_test)
forest_fpr, forest_tpr, forest_threshold = roc_curve(original_ytest, forest_test)
xgboost_fpr, xgboost_tpr, xgboost_threshold = roc_curve(original_ytest, xgboost_test)

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr, forest_fpr, forest_tpr, xgboost_fpr, xgboost_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Supervised Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, log_reg_test)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, knears_test)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, svc_test)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, tree_clf_test)))
    plt.plot(forest_fpr, forest_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, forest_test)))
    plt.plot(xgboost_fpr, xgboost_tpr, label='Xgboost Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, xgboost_test)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr, forest_fpr, forest_tpr, xgboost_fpr, xgboost_tpr)
supervised_fig = plt.gcf()
supervised_fig.savefig('supervised.eps', format='eps', dpi=1000)
plt.show()



# import pdb; pdb.set_trace()


#Unsupervised Models
#One-class SVM
nu = [0.1, 0.2, 0.5] 
gamma = [0.1, 0.01, 1]
best_model =0
for n in nu:
    for g in gamma:
        oc_svm = svm.OneClassSVM(nu = n, gamma = g)
        oc_svm = oc_svm.fit(original_Xtrain)
        oc_svm_test = oc_svm.predict(original_Xtest)
        oc_svm_test = np.where(oc_svm_test == -1, 1, 0)
        roc_score =roc_auc_score(original_ytest, oc_svm_test)
        if roc_score> best_model:
            best_model=roc_score
            bestn=n
            bestg=g
print("best nu:", n, "best gamma:", g, "one-class svm:",best_model,)


X_org_us = df.drop('Class', axis=1)
y_org_us = df['Class']

predictions_ae = np.zeros(len(X_org_us),np.float32)
labels_ae = np.zeros(len(y_org_us),np.uint8)
predictions_rbm = np.zeros(len(X_org_us),np.float32)
labels_rbm = np.zeros(len(y_org_us),np.uint8)

#do 5-fold cross validation
folds = 5
skf = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)
fold_num = 0
for train_index_us, test_index_us in skf.split(X_org_us, y_org_us):
	print('processing fold {:d} over {:}'.format(fold_num+1, folds))
	fold_num = fold_num + 1

    #Auto Encoder
	TEST_RATIO = 0.2
	df.sort_values('scaled_time', inplace = True)
	train_index_us = int((1-TEST_RATIO) * df.shape[0])
	X_train = df.iloc[:train_index_us, 1:-2].values
	Y_train = df.iloc[:train_index_us, -1].values

	X_test = df.iloc[train_index_us:, 1:-2].values
	Y_test = df.iloc[train_index_us:, -1].values
	print("Total train examples: {}, total fraud cases: {}, equal to {:.5f} of total cases. ".format(Y_train.shape[0], np.sum(X_train), np.sum(Y_train)/X_train.shape[0]))
	print("Total test examples: {}, total fraud cases: {}, equal to {:.5f} of total cases. ".format(X_test.shape[0], np.sum(Y_test), np.sum(Y_test)/Y_test.shape[0]))

# Parameters
	learning_rate = 0.001
	training_epochs = 10
	batch_size = 256
	display_step = 1

# Network Parameters
	n_hidden_1 = 15 # 1st layer num features
	#n_hidden_2 = 15 # 2nd layer num features
	n_input = X_train.shape[1] 
	data_dir = '.'

#Train the model
	X = tf.placeholder("float", [None, n_input])

	weights = {
    	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    #'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
	biases = {
    	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    	'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    #'decoder_b2': tf.Variable(tf.random_normal([n_input])),
	}


# Building the encoder
	def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   #biases['encoder_b2']))
		return layer_1


# Building the decoder
	def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                  # biases['decoder_b2']))
		return layer_1

# Construct model
	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

# Prediction
	y_pred = decoder_op
# Targets (Labels) are the input data.
	y_true = X

# Define batch mse
	batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)
# Define loss and optimizer, minimize the squared error
	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# TRAIN StARTS
	save_model = os.path.join(data_dir, 'temp_saved_model_1layer.ckpt')
	saver = tf.train.Saver()

# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		now = datetime.now()
		sess.run(init)
		total_batch = int(X_train.shape[0]/batch_size)
    # Training cycle
		for epoch in range(training_epochs):
        # Loop over all batches
			for i in range(total_batch):
				batch_idx = np.random.choice(X_train.shape[0], batch_size)
				batch_xs = X_train[batch_idx]
            # Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            
        # Display logs per epoch step
			if epoch % display_step == 0:
				train_batch_mse = sess.run(batch_mse, feed_dict={X: X_train})
				print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c), 
                  "Train auc=", "{:.6f}".format(roc_auc_score(Y_train, train_batch_mse)), 
                  "Time elapsed=", "{}".format(datetime.now() - now))

		print("Optimization Finished!")
    
		save_path = saver.save(sess, save_model)
		print("Model saved in file: %s" % save_path)

#Test the model
	save_model = os.path.join(data_dir, 'temp_saved_model_1layer.ckpt')
	saver = tf.train.Saver()

# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		now = datetime.now()
    
		saver.restore(sess, save_model)
    
		test_batch_mse = sess.run(batch_mse, feed_dict={X: X_test})
		predictions_ae[test_index_us] = test_batch_mse
		labels_ae[test_index_us] = Y_test
    
		print("Auto Encoder: {:.2f}".format(roc_auc_score(labels_ae[test_index_us], predictions_ae[test_index_us])))


#Restricted Boltzmann Machine
#RBM Model
	learning_rate = [0.001, 0.0001, 0.0005]
	num_epoch = [10, 20, 30]
	best_rbm = 0
	for l in learning_rate:
		for e in num_epoch:
			rbm = BernoulliRBM(X_train.shape[1], 10, visible_unit_type='gauss', main_dir='/Users/Jill/Documents/Internship_Project/fraud_transaction_detection /', model_name='rbm_model.ckpt', gibbs_sampling_steps=4, learning_rate=l, momentum = 0.95, batch_size=512, num_epochs=e, verbose=1)
			rbm.fit(X_train, validation_set = X_test)
			rbm_test = rbm.getFreeEnergy(X_test).reshape(-1)
			predictions_rbm[test_index_us] = rbm_test
			labels_rbm[test_index_us] = Y_test
			roc_sore = roc_auc_score(labels_rbm[test_index_us],  predictions_rbm[test_index_us])
			if roc_sore > best_rbm:
				best_rbm = roc_sore
				bestl = l
				beste = e
	print("best learning rate: ", l, "best num_epoch: ", e, "RBM: {:.2f}".format(roc_auc_score(labels_rbm[test_index_us], predictions_rbm[test_index_us])))



#ROC Curve of Unsupervised Models
oc_svm_fpr, oc_svm_tpr, svm_thresold = roc_curve(original_ytest, oc_svm_test)
ae_fpr, ae_tpr, ae_threshold = roc_curve(labels_ae[test_index_us], predictions_ae[test_index_us])
rbm_fpr, rbm_tpr, rbm_threshold = roc_curve(labels_rbm[test_index_us], predictions_rbm[test_index_us])


def graph_roc_curve_multiple(ae_fpr, ae_tpr, rbm_fpr, rbm_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Unsupervised Classifiers', fontsize=18)
    plt.plot(oc_svm_fpr, oc_svm_tpr, label='One-class SVM Classifier Score: {:.4f}'.format(roc_auc_score(original_ytest, oc_svm_test)))
    plt.plot(ae_fpr, ae_tpr, label='Auto Encoder Classifier Score: {:.4f}'.format(roc_auc_score(labels_ae[test_index_us], predictions_ae[test_index_us])))
    plt.plot(rbm_fpr, rbm_tpr, label='RBM Score: {:.4f}'.format(roc_auc_score(labels_rbm[test_index_us], predictions_rbm[test_index_us])))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(oc_svm_fpr, oc_svm_tpr, ae_fpr, ae_tpr, rbm_fpr, rbm_tpr)
unsupervised_fig = plt.gcf()
unsupervised_fig.savefig('unsupervised.eps', format='eps', dpi=1000)
plt.show()
