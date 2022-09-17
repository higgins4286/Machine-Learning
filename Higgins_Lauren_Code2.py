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
n_classes = T_binary.shape[1]

#Supervised classification dataset.
P_train, P_test, T_train, T_test = train_test_split(P_array, T_binary, test_size=0.3, 
                                                        train_size=0.7, random_state=0)

'''
Using PCA to reduce dimentionality before running 'linear', 'quadratic', 'diagLinear', 'diagQuadratic'.
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# standardizing data for effecient use of PCA
scaler = StandardScaler()
scaler.fit(P_train)
P_train_st = scaler.transform(P_train)
P_test_st = scaler.transform(P_test)

#print the dimentions of the input array
P_array.data.shape

#reducing dimentionality from 30 to 15
pca = PCA(n_components=15)

P_train_PCA = pca.fit_transform(P_train_st)
P_test_PCA = pca.transform(P_test_st)

###Put these values into the final plot
explained_ratio = pca.explained_variance_ratio_
explained_ratio_sum = explained_ratio.sum()

'''
Train the 4 learners: 'linear', 'quadratic', 'diagLinear', 'diagQuadratic'. 
Translation from MATLAB to Python:
    'linear'        --> LinearDiscriminantAnalysis()
    'diagLinear'    --> LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    'quadratic'     --> QuadraticDiscriminantAnalysis()
    'diagQuadratic' --> GaussianNB()
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

classifiers = [
    LinearDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    QuadraticDiscriminantAnalysis(),
    GaussianNB()]

classifiers_2 = [
    LinearDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    QuadraticDiscriminantAnalysis(),
    GaussianNB()]

key_names = ['lin', 'diagLin', 'quad', 'diagQuad']

### Empty dictionaries to store classification outcome and ROC metrics.

# Dictionaries for non-preprocessed
probs = {}
probst = {}   
score_og = {}
probs_pos = {}
probst_pos = {}
auc = {}
auct = {}

# Dictionaries for PCA preprocessed
probs_PCA = {}
probst_PCA = {}    
score_PCA = {}
probs_PCA_pos = {}
probst_PCA_pos = {}
auc_PCA = {}
auct_PCA = {}

i=0

for clf, clf_PCA, val in zip(classifiers, classifiers_2, key_names):
    '''
    Classify non-preprocessed data
    '''
    model = clf.fit(P_train, T_train)
    probs[val] = model.predict_proba(P_test)
    probst[val] = model.predict_proba(P_train)   
    score_og[val] = clf.score(P_test_st, T_test)
    
    # keep probabilities for the positive outcome only
    probs_pos[val] = probs[val][:, 1]
    probst_pos[val] = probst[val][:, 1]
    
    '''
    Classify PCA preprocessed data
    '''
    model_PCA = clf_PCA.fit(P_train_PCA, T_train)
    probs_PCA[val] = model_PCA.predict_proba(P_test_PCA)
    probst_PCA[val] = model_PCA.predict_proba(P_train_PCA)  
    score_PCA[val] = clf_PCA.score(P_test_PCA, T_test)     
    
    # keep probabilities for the positive outcome only
    probs_PCA_pos[val] = probs_PCA[val][:, 1]
    probst_PCA_pos[val] = probst_PCA[val][:, 1]
    
    '''
    Output for (a)ROC curves and (b)TP, FN, FP, and TN from the confusion matrix.
    '''
    # from sklearn.metrics import plot_confusion_matrix    
    # plot_confusion_matrix(clf, P_train, T_train)

    ##### (a)ROC curves #####

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt 

    ##calculate scores
    
    # score for non-preprocessed training and testing, respectively
    auc[val] = roc_auc_score(T_test, probs_pos[val])
    auct[val] = roc_auc_score(T_train, probst_pos[val])
    
    # score for PDA preprocessed training and testing, respectively
    auc_PCA[val] = roc_auc_score(T_test, probs_PCA_pos[val])
    auct_PCA[val] = roc_auc_score(T_train, probst_PCA_pos[val])   

    # summarize scores for plots
    s = '' + key_names[i] + ' : ROC Score=%.3f' % (auc[val])
    st = '' + key_names[i] + ' : ROC Score=%.3f' % (auct[val])
    
    s_PCA = '' + key_names[i] + ' : ROC Score=%.3f' % (auc_PCA[val])
    st_PCA = '' + key_names[i] + ' : ROC Score=%.3f' % (auct_PCA[val])
    
    # calculate roc curves
    
    fpr, tpr, _ = roc_curve(T_test, probs_pos[val])
    fprt, tprt, _ = roc_curve(T_train, probst_pos[val])
    
    fpr_PCA, tpr_PCA, _ = roc_curve(T_test, probs_PCA_pos[val])
    fprt_PCA, tprt_PCA, _ = roc_curve(T_train, probst_PCA_pos[val])
    
    # plot the roc curve for the models
    
    fig, ax = plt.subplots(2, 2, figsize=(20,20))
    
    ax[0, 0].plot(fpr, tpr, marker='.', 
                  label='Training Set '+s, 
                  linestyle='-.')
    ax[0, 1].plot(fprt, tprt, marker='.', label='Testing Set '+st, linestyle='--')
    
    ax[1, 0].plot(fpr_PCA, tpr_PCA, marker='.', 
                  label='PCA Training Set '+s_PCA  + '\n'+'Percent Variance Retained %.3f' % (explained_ratio_sum ), 
                  linestyle='-.')
    ax[1, 1].plot(fprt_PCA, tprt_PCA, marker='.', 
                  label='PCA Testing Set '+st_PCA  + '\n'+'Percent Variance Retained %.3f' % (explained_ratio_sum ), 
                  linestyle='--')
    
    # axis labels
    ax[0,0].set_xlabel('False Positive Rate', fontsize=14)
    ax[0,0].set_ylabel('True Positive Rate', fontsize=14)
    
    ax[0,1].set_xlabel('False Positive Rate', fontsize=14)
    ax[0,1].set_ylabel('True Positive Rate', fontsize=14)
    
    ax[1,0].set_xlabel('False Positive Rate', fontsize=14)
    ax[1,0].set_ylabel('True Positive Rate', fontsize=14)
    
    ax[1,1].set_xlabel('False Positive Rate', fontsize=14)
    ax[1,1].set_ylabel('True Positive Rate', fontsize=14)
    
    ax[0,0].set_title('No Dimentionality Reduction Training ROC', fontsize=18)
    ax[0,1].set_title('No Dimentionality Reduction Testing ROC', fontsize=18)
    ax[1,0].set_title('PCA Dimentionality Reduction Training ROC', fontsize=18)
    ax[1,1].set_title('PCA Dimentionality Reduction Testing ROC', fontsize=18)
    
    ax[0,0].legend(loc='lower right', fontsize='x-large')
    ax[0,1].legend(loc='lower right', fontsize='x-large')
    ax[1,0].legend(loc='lower right', fontsize='x-large')
    ax[1,1].legend(loc='lower right', fontsize='x-large')
    # plt.savefig('Higgins_Lauren_ROC_Curve_Case2_' + key_names[i] + '.pdf', bbox='tight')
    # plt.show()
    # plt.close(fig)


    ### Confusion Matricies
    from sklearn.metrics import plot_confusion_matrix
    
    def confusion(disc, data, targets, sample_set):
        class_names = ['benign', 'malignant']
        
        #Non-preprocessed data: Plot non-normalized and normalized confusion matrices
        titles_options = [("Task 1 " + sample_set + " set " + key_names[i] + " Normalized confusion matrix", 'true')]
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(disc, 
                                         data, 
                                         targets,                                    
                                          display_labels=class_names,
                                          cmap=plt.cm.Blues,
                                          normalize=normalize)
            disp.ax_.set_title(title)
        
            print(title)
            print(disp.confusion_matrix)
           
            return plt.savefig('Higgins_Lauren_Decision_Matrix_Case2_' + key_names[i] + '_' + sample_set + '.pdf', bbox='tight')
            plt.show()
            plt.close(fig)
        
    confusion(clf, P_test, T_test, 'Testing')
    confusion(clf, P_train, T_train, 'Training')
    confusion(clf_PCA, P_test_PCA, T_test, 'PCA Testing')
    confusion(clf_PCA, P_train_PCA, T_train, 'PCA Training')    
            
    i += 1
#end