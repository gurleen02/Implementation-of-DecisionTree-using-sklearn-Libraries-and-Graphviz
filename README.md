# Implementation-of-DecisionTree-using-sklearn-Libraries-and-Graphviz

Decision Trees are an easy-to-understand visualization technique and are used for fast deployment into production. It is using a binary tree graph (each node has two children) to assign for each data sample a target value. 

## Data Source ##

    The iris dataset from scikit learn datasets has been used and fitting has been performed using the default attributes.
    
## Step 1: Importing the modules and libraries ##

    from matplotlib import pyplot as plt
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn import tree
    
## Step 2: Preparing the Data ##

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

## Step 3: Fitting the model using the default hyperparameters ##
    
    clf = DecisionTreeClassifier(random_state=1234)
    model = clf.fit(X, y)
    
## Step 4: Performing textual representation of the features and classes ##
    
    text_representation = tree.export_text(clf)
    print(text_representation)
    
## Step 5: Plotting the decision tree ##

    fig = plt.figure(figsize=(25,20))_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)
                   
## Step 6: Plotting the decision tree ##
    
    
