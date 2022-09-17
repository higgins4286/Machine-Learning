'''
Loading the two data sets into my code and converting the dictionaries into numpy arrays.
'''
from scipy.io import loadmat

#Load datasets.
P = loadmat('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/BCdata/P.mat', mat_dtype=True)
T = loadmat('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/BCdata/T.mat', mat_dtype=True)

#Convert dictionaries into numpy arrays.

P_array = P['P'].T
T_array = T['T'].T

'''
Create a supervised classification dataset and arrange the target data between P and T. 

For task 1 randomly 70% of the data with be used for 'training' 
For task 2 randomly 40% of the data will be used for 'training'.
This step automatically labels the data as two different classes.
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Binarize the output
T_binary = label_binarize(T_array, classes=[0, 1])
n_classes = T_binary.shape[1]

#Supervised classification dataset.
P_train1, P_test1, T_train1, T_test1 = train_test_split(P_array, T_binary, test_size=0.3, 
                                                        train_size=0.7, random_state=0)
P_train2, P_test2, T_train2, T_test2 = train_test_split(P_array, T_binary, test_size=0.6, 
                                                        train_size=0.4, random_state=0)


'''
Train the Fisher LDA for both task 1 (model1) and task 2 (model2) and 
use testing data to validate results. 
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

model1 = LDA()
task1 = model1.fit(P_train1, T_train1)
probs1 = task1.predict_proba(P_test1)
probs1t = task1.predict_proba(P_train1)

model2 = LDA()
task2 = model2.fit(P_train2, T_train2)
probs2 = task2.predict_proba(P_test2)
probs2t = task2.predict_proba(P_train2)

'''
Output for (a)ROC curves and (b)TP, FN, FP, and TN from the confusion matrix.
'''

##### (a)ROC curves #####

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 

# keep probabilities for the positive outcome only
task1_probs = probs1[:, 1]
task2_probs = probs2[:, 1]

task1_probst = probs1t[:, 1]
task2_probst = probs2t[:, 1]


# calculate scores

#score for training
task1_auc = roc_auc_score(T_test1, task1_probs)
task2_auc = roc_auc_score(T_test2, task2_probs)

#score for testing
task1_auct = roc_auc_score(T_train1, task1_probst)
task2_auct = roc_auc_score(T_train2, task2_probst)

# summarize scores
s1 = ' LDA: ROC AUC=%.3f' % (task1_auc)
s2 = ' LDA: ROC AUC=%.3f' % (task2_auc)

s1t = ' LDA: ROC AUC=%.3f' % (task1_auct)
s2t = ' LDA: ROC AUC=%.3f' % (task2_auct)
# calculate roc curves
task1_fpr, task1_tpr, _ = roc_curve(T_test1, task1_probs)
task2_fpr, task2_tpr, _ = roc_curve(T_test2, task2_probs)

task1_fprt, task1_tprt, _ = roc_curve(T_train1, task1_probst)
task2_fprt, task2_tprt, _ = roc_curve(T_train2, task2_probst)

print(task1_fpr)

# plot the roc curve for the model

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

ax1.plot(task1_fpr, task1_tpr, marker='.', label='Task 1: 70/30 Split'+s1, linestyle='-.')
ax1.plot(task2_fpr, task2_tpr, marker='.', label='Task 2: 40/60 Split'+s2, linestyle=':')
ax2.plot(task1_fprt, task1_tprt, marker='.', label='Task 1: 70/30 Split'+s1t, linestyle='--')
ax2.plot(task2_fprt, task2_tprt, marker='.', label='Task 2: 40/60 Split'+s2t, alpha=0.5)

# axis labels
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')

ax1.set_title('Training ROC')
ax2.set_title('Testing ROC')

ax1.legend()
ax2.legend()
#plt.savefig('Higgins_Lauren_ROC_Curve.pdf', bbox='tight')
plt.show()

from sklearn.metrics import plot_confusion_matrix

class_names = ['benign', 'malignant']

## Task 1

#Testing data: Plot non-normalized confusion matrix
titles_options = [("Task 1 Testing set Confusion matrix, without normalization", None),
                  ("Task 1 Testing set Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(task1, P_test1, T_test1,
                                  display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()

#Training data: Plot non-normalized confusion matrix
titles_options = [("Task 1 Training set Confusion matrix, without normalization", None),
                  ("Task 1 Training set Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(task1, P_train1, T_train1,
                                  display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()

##Task 2
#Testing data: Plot non-normalized confusion matrix
titles_options = [("Task 2 Testing set Confusion matrix, without normalization", None),
                  ("Task 2 Testing set Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(task2, P_test2, T_test2,
                                  display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()


#Training data: Plot non-normalized confusion matrix
titles_options = [("Task 2 Training set Confusion matrix, without normalization", None),
                  ("Task 2 Training set Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(task2, P_train2, T_train2,
                                  display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()

###end