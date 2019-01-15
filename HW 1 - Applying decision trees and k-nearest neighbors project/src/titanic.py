"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        #print(majority_val)      # 0.0
        #print(Counter(y).most_common(1)[0])   # (0.0, 424)
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        random_val = Counter(y).most_common(2)      # dictionary: [(0.0, 424), (1.0, 288)]
        total = random_val[0][1] + random_val[1][1]  # 712
        #print(random_val, total)
        self.probabilities_ = random_val[0][1] / total    # 0.485
        #self.probabilities_ = random_val[1][1] / total   # 0.527
        
        #if random_val[0][0] == 0:
            #self.probabilities_ = random_val[0][1] / total
        #else:
            #self.probabilities_ = random_val[1][1] / total
                
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        #print(X.shape)   # (712, 7)
        y = np.random.choice(2, X.shape[0], p=[self.probabilities_, 1-self.probabilities_])
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_err = 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        test_err = 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        train_error += train_err
        test_error += test_err
    train_error = train_error / ntrials
    test_error = test_error / ntrials
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    #print('Plotting...')
    #for i in range(d) :
        #plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)      # 0.404
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    ranClf = RandomClassifier()
    ranClf.fit(X, y)
    ran_y_pred = ranClf.predict(X)
    ran_train_error = 1 - metrics.accuracy_score(y, ran_y_pred, normalize=True)
    print('\t-- training error: %.3f' % ran_train_error)      # 0.485
    
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    
    deciTreeClf = DecisionTreeClassifier(criterion="entropy")
    deciTreeClf.fit(X, y)
    dT_y_pred = deciTreeClf.predict(X)
    dT_train_err = 1 - metrics.accuracy_score(y, dT_y_pred, normalize=True)
    print('\t-- training error: %.3f' % dT_train_err)       # 0.014
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    
    kNN_3 = KNeighborsClassifier(n_neighbors=3)
    kNN_3.fit(X,y)
    kNN_3_y_pred = kNN_3.predict(X)
    kNN_3_tranErr = 1 - metrics.accuracy_score(y, kNN_3_y_pred, normalize=True)
    print('\t-- training error for k = 3: %.3f' % kNN_3_tranErr)       # 0.167
    
    kNN_5 = KNeighborsClassifier(n_neighbors=5)
    kNN_5.fit(X,y)
    kNN_5_y_pred = kNN_5.predict(X)
    kNN_5_tranErr = 1 - metrics.accuracy_score(y, kNN_5_y_pred, normalize=True)
    print('\t-- training error for k = 5: %.3f' % kNN_5_tranErr)       # 0.201
    
    kNN_7 = KNeighborsClassifier(n_neighbors=7)
    kNN_7.fit(X,y)
    kNN_7_y_pred = kNN_7.predict(X)
    kNN_7_tranErr = 1 - metrics.accuracy_score(y, kNN_7_y_pred, normalize=True)
    print('\t-- training error for k = 7: %.3f' % kNN_7_tranErr)       # 0.240
        
    ### ========== TODO : END ========== ###
    
    
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
        
    majorClf = MajorityVoteClassifier()
    randomClf = RandomClassifier()
    dtClf = DecisionTreeClassifier(criterion='entropy')
    kNN_5_Clf = KNeighborsClassifier(n_neighbors=5)
    clfList = [majorClf, randomClf, dtClf, kNN_5_Clf]
    for clf in clfList: 
        train_error, test_error = error(clf, X, y)
        print("Average results of {}:\n\t-- training error: {:.3f}, test error: {:.3f}".format(clf.__class__.__name__, train_error, test_error))
    
    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    
    kList = []       # k = 1, 3, 5, 7, ...
    errList = []     # error for k = 1, 3, 5, 7, ...
    for k in range(1, 50, 2): 
        clf_estimator = KNeighborsClassifier(n_neighbors=k)
        accuracy = cross_val_score(clf_estimator, X, y, cv=10)   # 10-fold
        #print(accuracy)       # accuracy rate are 10 values (as a numpy array) from 10-fold cv
        #[ 0.65277778  0.58333333  0.69444444  0.68055556  0.69014085  0.67605634
        #  0.61971831  0.69014085  0.62857143  0.75714286]        
        errors = 1-np.mean(accuracy)
        #print(errors)         # err_rate = 1 - mean(accuracy above); 0.332711826515
        kList.append(k)              # appending k = 1, 3, 5, 7, ... to do graph
        errList.append(errors)      # appending the error to do graph
        
    plt.clf()       # clear the figure before picture it
    plt.plot(kList, errList)
    plt.xlabel('k, number of neighbours')
    plt.ylabel('y, error rate')
    plt.savefig("4f_pic.pdf")
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    
    kList = []
    trainErr = []
    testErr = []
    for k in range(1, 20): 
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=k)
        train_error, test_error = error(clf, X, y)
        kList.append(k)
        trainErr.append(train_error)     # obtain train error array
        testErr.append(test_error)       # obtain test error array
    plt.clf()
    plt.plot(kList, testErr, 'gs-', label="test data")
    plt.plot(kList, trainErr, 'r^-', label="training data")
    plt.legend()
    plt.xlabel('k, decision tree depth')
    plt.ylabel('y, error rate')
    plt.savefig('4g_pic.pdf')
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    index = []
    dtTrainErrs = []
    dtTestErrs = []
    knnTrainErrs = []
    knnTestErrs = []
    for i in range(1, 11):
        dtTrainErrSum = 0
        dtTestErrSum = 0
        knnTrainErrSum = 0
        knnTestErrSum = 0
        index.append(i * 0.1)      # i th propotion of size
        for k in range(100):
            X_train2, y_train2 = X_train, y_train
            if i != 10:
                X_train2, _, y_train2, _ = train_test_split(X_train, y_train, test_size=(1-i*0.1), random_state=k)
            dtClf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            knnClf = KNeighborsClassifier(n_neighbors=7, p=2)

            dtClf.fit(X_train2, y_train2)
            dt_y_train_pred = dtClf.predict(X_train2)
            train_error1 = 1 - metrics.accuracy_score(y_train2, dt_y_train_pred, normalize=True)
            dtTrainErrSum += train_error1
            
            dt_y_test_pred = dtClf.predict(X_test)
            test_error1 = 1 - metrics.accuracy_score(y_test, dt_y_test_pred, normalize=True)
            dtTestErrSum += test_error1

            knnClf.fit(X_train2, y_train2)
            knn_y_train_pred = knnClf.predict(X_train2)
            train_error2 = 1 - metrics.accuracy_score(y_train2, knn_y_train_pred, normalize=True)
            knnTrainErrSum += train_error2
            
            knn_y_test_pred = knnClf.predict(X_test)
            test_error2 = 1 - metrics.accuracy_score(y_test, knn_y_test_pred, normalize=True)
            knnTestErrSum += test_error2
        dtTrainErrs.append(dtTrainErrSum/100)
        dtTestErrs.append(dtTestErrSum/100)
        knnTrainErrs.append(knnTrainErrSum/100)
        knnTestErrs.append(knnTestErrSum/100)

    plt.clf()
    plt.plot(index, dtTrainErrs, 'ro-', \
            index, dtTestErrs, 'bo-', \
            index, knnTrainErrs, 'r^-', \
            index, knnTestErrs, 'b^-')
    red_circle = mpl.lines.Line2D([], [], color='r', marker='o', label='training error for DT')
    green_circle = mpl.lines.Line2D([], [], color='b', marker='o', label='test error for DT')
    red_tri = mpl.lines.Line2D([], [], color='r', marker='^', label='training error for KNN')
    green_tri = mpl.lines.Line2D([], [], color='b', marker='^', label='test error for KNN')
    #plt.legend(handles=[red_circle, green_circle, red_tri, green_tri])
    plt.xlabel('size of 90% training set')
    plt.ylabel('training rate/test rate')
    plt.savefig('4h_pic.pdf')
    
    
    
    
    
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
