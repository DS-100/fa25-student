import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def words_in_texts(words, texts):
    '''
    Args:
        words (list): words to find.
        texts (Series): strings to search in.
    
    Returns:
        A 2D NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts, and p is the number of words.
    '''
    indicator_array = 1 * np.array([texts.str.contains(word) for word in words]).T
    return indicator_array

def compute_CV_error(X_train, Y_train, folds):
    '''
    Split the training data into `k` subsets.
    For each subset, 
        - Fit a model holding out that subset.
        - Compute the MSE on that subset (the validation set).
    Return a list of `k` accuracies.

    Args:
        X_train (numpy array): Training data design matrix.
        Y_train (numpy array): Label.
    
    Return:
         A list of `k` accuracies.
    '''
    model = LogisticRegression(solver = 'lbfgs')
    kf = KFold(n_splits=folds)
    
    validation_accuracies = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # Split the data
        split_X_train, split_X_valid = X_train[train_idx], X_train[valid_idx]
        split_Y_train, split_Y_valid = Y_train[train_idx], Y_train[valid_idx]
      
      
        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)
        
        accuracy = np.mean(model.predict(split_X_valid) == split_Y_valid)
        # END SOLUTION

        validation_accuracies.append(accuracy)

    return validation_accuracies