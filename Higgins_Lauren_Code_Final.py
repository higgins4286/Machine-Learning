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
This step automatically labels the data as two different classes.
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Binarize the output
T_binary = label_binarize(T_array, classes=[0, 1])

#Supervised classification dataset.
P_train, P_test, T_train, T_test = train_test_split(P_array, T_binary, test_size=0.3, 
                                                        train_size=0.7, random_state=66)

'''
Standardize the input data.
'''

from sklearn.preprocessing import StandardScaler

# standardizing data for effecient use in SVM
scaler = StandardScaler()
scaler.fit(P_train)
P_train_st = scaler.transform(P_train)
P_test_st = scaler.transform(P_test)

'''
Setting up my loop that will go through the low bias, high variance classifiers. They are:
    
    Fisher LDA                                                         --> LinearDiscriminantAnalysis()
    Quadratic Discriminant Analysis                                    --> QuadraticDiscriminantAnalysis()
    Gaussian SVM: Radial Basis Function with low s^2 with hard margins --> svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True)
    3rd Order Polynomial SVM with hard margins                         --> vm.SVC(C=1, kernel='poly', degree=3, gamma='auto', probability=True)

Before the classification loop, I will utilize the sklearn.feature_selection package to utilize 
forward/backward feature selection to create a subset of features for my four classifiers to 
use in classification.

BONUS: Within the loop, I will also use stack generalization to create a meta learner.
    
'''

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm

classifiers = [
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True),
    svm.SVC(C=1, kernel='poly', degree=3, gamma='auto', probability=True)]

class_type = ['FLDA', 'QuadraticDiscriminantAnalysis', 'RBF SVM', '3rd order SVM']

classifiers2 = [
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True),
    svm.SVC(C=1, kernel='poly', degree=3, gamma='auto', probability=True)]

class_type2 = ['FLDA', 'QuadraticDiscriminantAnalysis', 'RBF SVM', '3rd order SVM']

classifiers3 = [
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True),
    svm.SVC(C=1, kernel='poly', degree=3, gamma='auto', probability=True)]

# Dictionaries for non-preprocessed
model = {}
probs = {}
probst = {}   
probs_pos = {}
probst_pos = {}

auc = {}
auct = {}

fpr= {}
tpr = {}
fprt = {}
tprt = {}

s = {}
st = {}

# Dictionaries for sfs preprocessed
model_sfs = {}
probs_sfs = {}
probst_sfs = {}    
probs_sfs_pos = {}
probst_sfs_pos = {}

auc_sfs = {}
auct_sfs = {}

fpr_sfs = {}
tpr_sfs = {}
fprt_sfs = {}
tprt_sfs = {}

s_sfs = {}
st_sfs = {}

# Dictionaries for fusion classification
model_fus = {}
probs_fus = {}
probst_fus = {}
probs_fus_pos = {}
probst_fus_pos = {}

auc_fus = {}
auct_fus = {}

fpr_fus = {}
tpr_fus = {}
fprt_fus = {}
tprt_fus = {}

s_fus = {}
st_fus = {}

'''

Preprocessing using sequential forward feature selection. The python equivalent 
to 'sequentialfs' is 'SequentialFeatureSelector'.

'''

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

sfs = SequentialFeatureSelector(knn, n_features_to_select=15, direction='backward')

P_train_sfs = sfs.fit_transform(P_train_st, T_train)
P_test_sfs = sfs.transform(P_test_st)

'''

Training the classifiers with original data and feature selected data.

'''

i = 0

for clf, clf_sfs, name, name2 in zip(classifiers, classifiers2, class_type, class_type2):
    
    '''
    Four low bias, high variance classifiers on original data.
    '''
    
    # training the models
    model[name] = clf.fit(P_train_st, T_train)
    probs[name] = model[name].predict_proba(P_test_st)
    probst[name] = model[name].predict_proba(P_train_st)   
    
    # keep probabilities for the positive outcome only
    probs_pos[name] = probs[name][:, 1]
    probst_pos[name] = probst[name][:, 1] 
    
    '''
    Four additional classifiers using feature selected data.
    '''
    model_sfs[name2] = clf_sfs.fit(P_train_sfs, T_train)
    probs_sfs[name2] = model_sfs[name2].predict_proba(P_test_sfs)
    probst_sfs[name2] = model_sfs[name2].predict_proba(P_train_sfs)     
    
    # keep probabilities for the positive outcome only
    probs_sfs_pos[name2] = probs_sfs[name2][:, 1]
    probst_sfs_pos[name2] = probst_sfs[name2][:, 1]
    
    '''
    **BONUS**
    
    Stack generalization to train a meta learner of your choice
    
    '''
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    estimators = [
      ('lda', LinearDiscriminantAnalysis()),
      ('qda', QuadraticDiscriminantAnalysis()),
      ('rbf', svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True)),
      ('poly', svm.SVC(C=1, kernel='poly', degree=3, gamma='auto', probability=True))]
    
    # training stack generalization
    clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    model_stack = clf_stack.fit(P_train_st, T_train)
    probs_stack = clf_stack.predict_proba(P_test_st)
    probst_stack = clf_stack.predict_proba(P_train_st)
    
    # keep probabilities for the positive outcome only
    probs_stack_pos = probs_stack[:, 1]
    probst_stack_pos = probst_stack[:, 1]    

    '''
    
    AUC Scores for my four base classifiers both with original data and feature selected data and
    my stack generalization scores.
    
    
    '''
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    # score
    auc[name] = roc_auc_score(T_test, probs_pos[name])   
    auct[name] = roc_auc_score(T_train, probst_pos[name])

    
    auc_sfs[name2] = roc_auc_score(T_test, probs_sfs_pos[name]) 
    auct_sfs[name2] = roc_auc_score(T_train, probst_sfs_pos[name])
    
    auc_stack = roc_auc_score(T_test, probs_stack_pos)
    auct_stack = roc_auc_score(T_train, probst_stack_pos)
    
    # summarize scores for plots
    s[name] = '' + class_type[i] + ': ROC Score=%.5f' % (auc[name])       
    st[name] = '' + class_type[i] + ': ROC Score=%.5f' % (auct[name])
    
    s_sfs[name2] = '' + class_type2[i] + ' Backward FS: ROC Score=%.5f' % (auc_sfs[name])
    st_sfs[name2] = '' + class_type2[i] + ' Backward FS: ROC Score=%.5f' % (auct_sfs[name])
           
    st_stack = 'BONUS Stack Generalization: ROC Score=%.5f' % (auct_stack)
    
    # calculate roc curves
    fpr[name], tpr[name], _ = roc_curve(T_test, probs_pos[name])
    fprt[name], tprt[name], _ = roc_curve(T_train, probst_pos[name])

    fpr_sfs[name], tpr_sfs[name], _ = roc_curve(T_test, probs_sfs_pos[name])
    fprt_sfs[name], tprt_sfs[name], _ = roc_curve(T_train, probst_sfs_pos[name])
    
    fpr_stack, tpr_stack, _ = roc_curve(T_test, probs_stack_pos)
    fprt_stack, tprt_stack, _ = roc_curve(T_train, probst_stack_pos)
    
'''

Fusion Ensembles:

    Per table 18.2, I will choose max and sum for the best fusion classifier. 
    
    I will use the python modue VotingClassifier to use the sum fusion method.

'''

eclf = {}

for name3, name4 in zip(model, model_sfs):

    # Create my sum fusion classifiers
    from sklearn.ensemble import VotingClassifier
    
    eclf1 = VotingClassifier(estimators=[('og_1', model['FLDA']), 
                                         ('og2_1', model['QuadraticDiscriminantAnalysis']),
                                          ('pre_1', model_sfs['FLDA']),
                                          ('pre2_1', model_sfs['QuadraticDiscriminantAnalysis']),
                                          ('stack_1', clf_stack)], 
                              voting='soft')
    eclf2 = VotingClassifier(estimators=[('og_2', model['3rd order SVM']), 
                                         ('og2_2', model['RBF SVM']),
                                         ('pre_2', model_sfs['3rd order SVM']),
                                          ('pre2_2', model_sfs['RBF SVM']),
                                          ('pre3_2', model_sfs['FLDA']),
                                          ('stack_2', clf_stack)], 
                              voting='soft')
    eclf3 = VotingClassifier(estimators=[('og_3', model['RBF SVM']), 
                                         ('og2_3', model['3rd order SVM']),
                                          ('pre_3', model_sfs['RBF SVM']),
                                          ('pre2_3', model_sfs['FLDA']),
                                          ('stack_3', clf_stack)], 
                              voting='soft')
    eclf4 = VotingClassifier(estimators=[('og_4', model['RBF SVM']), 
                                         ('og2_4', model['3rd order SVM']),
                                          ('pre_4', model_sfs['RBF SVM']),
                                          ('pre2_4', model_sfs['3rd order SVM']),
                                          ('stack_4', clf_stack)], 
                              voting='soft')
    
    eclf = {'FLDA': eclf1, 
            'QuadraticDiscriminantAnalysis': eclf2, 
            'RBF SVM': eclf3, 
            '3rd order SVM': eclf4
            }

'''

Caclulating AUC scores for the sum fusion results. 

''' 
for vers in eclf:

    model_fus[vers] = eclf[vers].fit(P_train_st, T_train)
    probs_fus[vers] = model_fus[vers].predict_proba(P_test_st)
    probst_fus[vers] = model_fus[vers].predict_proba(P_train_st)
    
    probs_fus_pos[vers] = probs_fus[vers][:, 1]
    probst_fus_pos[vers] = probst_fus[vers][:, 1]   
    

    # score
    auc_fus[vers] = roc_auc_score(T_test, probs_fus_pos[vers])   
    auct_fus[vers] = roc_auc_score(T_train, probst_fus_pos[vers])
    
    # summarize scores for plots
    s_fus[vers] = 'Sum Fusion: ROC Score=%.5f' % (auc_fus[vers])       
    st_fus[vers] = 'Sum Fusion: ROC Score=%.5f' % (auct_fus[vers])
    
    # calculate roc curves
    fpr_fus[vers], tpr_fus[vers], _ = roc_curve(T_test, probs_fus_pos[vers])
    fprt_fus[vers], tprt_fus[vers], _ = roc_curve(T_train, probst_fus_pos[vers])

'''

Plotting the top three fusion results and compairing them to the original ROC for all classifiers.

'''
i=0

eclf_type = ['eclf1', 'eclf2', 'eclf3', 'eclf4']

for key, key, key, key, key, key, key, key, key in zip(fpr.keys(), tpr.keys(), s.keys(), fpr_sfs.keys(), tpr_sfs.keys(), 
                         s_sfs.keys(), fprt_fus.keys(), tpr_fus.keys(), s_fus.keys()):
        
    # plot the roc curve for the models
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    
    ax1.plot(fpr[key], tpr[key], marker='.', label='Training '+ s[key], linestyle='-.')           
    ax1.plot(fpr_sfs[key], tpr_sfs[key], marker='.', label='Training '+s_sfs[key], 
             linestyle='-')     
    ax1.plot(fpr_fus['QuadraticDiscriminantAnalysis'], 
             tpr_fus['QuadraticDiscriminantAnalysis'], marker='o', 
             label='Training V4 '+s_fus['QuadraticDiscriminantAnalysis'], 
              linestyle='-')
    
    ax2.plot(fprt[key], tprt[key], marker='.', label='Testing '+st[key], linestyle='-.')
    ax2.plot(fprt_sfs[key], tprt_sfs[key], marker='.', label='Testing '+st_sfs[key], 
             linestyle='-')
    ax2.plot(fprt_stack, tprt_stack, marker='*', label='Testing '+st_stack, linestyle=':') 
    ax2.plot(fprt_fus['QuadraticDiscriminantAnalysis'], 
             tprt_fus['QuadraticDiscriminantAnalysis'], marker='o', 
             label='Testing V4 '+st_fus['QuadraticDiscriminantAnalysis'], 
               linestyle='-')
    
    # axis labels
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    
    ax2.set_xlabel('False Positive Rate', fontsize=14)
    ax2.set_ylabel('True Positive Rate', fontsize=14)
    
    ax1.set_title(class_type[i] + ' Training ROC', fontsize=18)
    ax2.set_title(class_type[i] + ' Testing ROC', fontsize=18)
    
    ax1.legend(loc='lower right', fontsize='x-large')
    ax2.legend(loc='lower right', fontsize='x-large')
    
    plt.savefig('Higgins_Lauren_ROC_Curves_eclf2_' + class_type[i] + '.pdf', bbox='tight')
    plt.show()
    plt.close(fig)
    
    i += 1    
    
# end