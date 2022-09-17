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

# standardizing data for effecient use of PCA
scaler = StandardScaler()
scaler.fit(P_train)
P_train_st = scaler.transform(P_train)
P_test_st = scaler.transform(P_test)

from sklearn.preprocessing import StandardScaler
from sklearn import svm

'''
Task 1

Train the 4 SVM: RBF, linear, 2nd order polynomial, 3rd order polynomial. 
Translation from MATLAB to Python:
    RBF                      --> SVC(‘rbf’, gamma='auto')
    linear                   --> SVC(‘linear’, gamma='auto')
    2nd order polynomial     --> SVC(‘poly’, degree=2, gamma='auto')
    3rd order polynomial     --> SVC(‘poly’, degree=3, gamma='auto')
    'Standardize', true      --> StandardScaler()     
'''

# Declare svm types, degree, and labes for saving and titling the outputs.

kernel = ['rbf', 'linear', 'poly', 'poly']
degree = [3, 3, 2, 3]
order_lab = ['', '', '2', '3']

'''
Classify  kernels
'''

# fit the SVMs
for kern, order, val in zip(kernel, degree, order_lab):
                                                 
    '''
    Classify Task 1 kernels
    '''
    
    # Training
    clf = svm.SVC(kernel=kern, degree=order, gamma='auto')
    clf.fit(P_train_st, T_train)
    T_pred = clf.predict(P_test_st)
    
    '''
    Output for TP, FN, FP, and TN from the confusion matrix.
    '''

    ### Confusion Matricies
    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    
    def confusion(svm_type, data, targets, sample_set, file_name, color, norm, norms):
        class_names = ['benign', 'malignant']

        
        #Non-preprocessed data: Plot non-normalized and normalized confusion matrices
        titles_options = [(sample_set + " set " + kern + val + norms + "confusion matrix", 
                                norm)]
        
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(svm_type, 
                                          data, 
                                          targets,                                    
                                          display_labels=class_names,
                                          cmap=color,
                                          normalize=norm)
            disp.ax_.set_title(title)
        
            print(title)
            print(disp.confusion_matrix)
            print(order_lab)
            return plt.savefig('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/Proj_3/Task_1_Plots/Higgins_Lauren_Decision_Matrix_Case3_TF' + kern + val + '_' + file_name + '.pdf', 
                                bbox='tight')
            plt.show()
            plt.close(disp)  
        
    confusion(clf, P_test_st, T_test, 'Task 1 Testing', 'Task_1_Testing', plt.cm.Blues, 'true', 'Normalized ')
    confusion(clf, P_train_st, T_train, 'Task 1 Training', 'Task_1_Training', plt.cm.Blues, 'true', 'Normalized ')
    confusion(clf, P_test_st, T_test, 'Task 1 Testing', 'Task_1_Testing', plt.cm.Greens, None, '')
    confusion(clf, P_train_st, T_train, 'Task 1 Training', 'Task_1_Training', plt.cm.Greens, None, '')

'''
Task 2

Change the 'box constraints' to get better testing results.
Translation from MATLAB to Python:
    'box constraint' --> 'C'

'''

# Declare svm types, degree, and labes for saving and titling the outputs

kernel_task2 = ['rbf', 'linear', 'poly', 'poly']
degree_task2 = [3, 3, 2, 3]
order_lab2 = ['', '', '2', '3']  
  
# fit the SVMs
for kern2, order2, val2 in zip(kernel_task2, degree_task2, order_lab2):
    
    box_constraints = [100, 10, 0.1]
    
    for box2 in box_constraints:
    
        '''
        Classify Task 2 kernels
        '''
        
        # Training set
        clf_task2 = svm.SVC(C=box2, kernel=kern2, degree=order2, gamma='auto')
        clf_task2.fit(P_train_st, T_train)
        T_pred2 = clf_task2.predict(P_test_st)
    
    
        '''
        Output for TP, FN, FP, and TN from the confusion matrix.
        '''
    
        ### Confusion Matricies 
    
        
        def confusion(svm_type2, data2, targets2, sample_set2, file_name2, color2, norm2, norms2):
            class_names2 = ['benign', 'malignant']
    
            
            #Non-preprocessed data: Plot non-normalized and normalized confusion matrices
            titles_options2 = [(sample_set2 + " set " + kern2 + val2 + " %.3f " % box2 + norms2 + "confusion matrix", 
                                norm2)]
            
            for title2, normalize2 in titles_options2:
                disp2 = plot_confusion_matrix(svm_type2, 
                                              data2, 
                                              targets2,                                    
                                              display_labels=class_names2,
                                              cmap=color2,
                                              normalize=norm2)
                disp2.ax_.set_title(title2)
            
                print(title2)
                print(disp2.confusion_matrix)
                print(order_lab)
                return plt.savefig('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/Proj_3/Task_2_Plots/Higgins_Lauren_Decision_Matrix_Case3_' + kern2 + val2 + norms2 + "_%.3f " % box2 + '_' + file_name2 + '.pdf', 
                                    bbox='tight')
                plt.show()
                plt.close(disp2)  
        
        confusion(clf_task2, P_test_st, T_test, 'Task 2 Testing', 'Task_2_Testing', plt.cm.Purples, 'true', 'Normalized ')
        confusion(clf_task2, P_train_st, T_train, 'Task 2 Training', 'Task_2_Training', plt.cm.Purples, 'true', 'Normalized ')  
        confusion(clf_task2, P_test_st, T_test, 'Task 2 Testing', 'Task_2_Testing', plt.cm.Oranges, None, '')
        confusion(clf_task2, P_train_st, T_train, 'Task 2 Training', 'Task_2_Training', plt.cm.Oranges, None, '') 

# #end