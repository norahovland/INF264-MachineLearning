import numpy as np
import pandas as pd
from math import log2
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
"""Nora Hobæk Hovland----------------INF264-------------Project1-------------------------------"""

#preprocessing of the abelone dataset:

filename = 'INF264-MachineLearning/INF264/abalone.data'
data = np.loadtxt(filename, dtype=str, delimiter=",")

#columnNames to make the printing of the tree nicer
columnNamesStart=np.array(["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked Weight",
"Viscera weight", "Shell weight"])

#assign integers to the letters in the first column
for i in range(data.shape[0]):
    if data[i][0]=="M":
        data[i][0]=0
    elif data[i][0]=="F":
        data[i][0]=1
    else:
        data[i][0]=2

# y is the last column of the input
y = data[:, -1]  
# X is all except the last column
X=np.delete(data, data.shape[1]-1, 1) 

#read data as floats
X = X.astype(np.float)  
y = y.astype(np.float)

#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, shuffle=False)

"""-----------------------------------------------------------------------------------------"""

# make thresholds for all the colums of X
def make_thresholdsX(X: np.ndarray):
    thresholdList = []
    for i in range(X.shape[1]):
        threshold = np.average(X[:, i]) 
        thresholdList.append(threshold)
             
    return thresholdList


#calculate entropy for the labels
def entropy(y):
    probY=probability(y)
    entropy = 0

    for prob in probY:
        entropy = entropy + prob * log2(prob)
    return -entropy


# find probability of the labels
def probability(y):
    counts = np.unique(y, return_counts=True)[1]
    return np.array(counts)/len(y)

#calculate gini index
def giniIndexLabel(probY):
    giniSum = 0
    for prob in probY:
        giniSum = giniSum + prob * prob    
    return 1-giniSum


def countSmallerAndBigger(X, thresholdList):
    smallerX={}
    biggerX={}

    for col in range(X.shape[1]):
        smaller=0
        bigger=0
        #count number of elements under and above threshold
        for row in range(X.shape[0]):
            if (X[row][col]<thresholdList[col]):
                smaller=smaller+1
            else:
                bigger=bigger+1
        
        #add to dicts
        smallerX[col] = smaller
        biggerX[col] = bigger

    return smallerX, biggerX



#choose which column to split by finding entropies for all columns
def chooseSplitColumnWithEntropy(X, y, thresholdList):
    #number of elements above and under thresholds in each column of X
    smallerX, biggerX = countSmallerAndBigger (X, thresholdList)
 
    #find conditional entropies for each column
    entropiesLabelGivenColumn = {}

    #(label|value)
    for col in range(X.shape[1]):
        #probability of values above threshold
        probXBig = biggerX[col]/(biggerX[col]+smallerX[col])
        #probability of values smaller than threshold
        probXSmall = smallerX[col]/(biggerX[col]+smallerX[col])

        #split into set of labels given value above and under threshold
        yGivenValueBig = []  
        yGivenValueSmall = [] 

        for row in range(X.shape[0]):
            if(X[row][col] >= thresholdList[col]):
                yGivenValueBig.append(y[row])
            else:
                yGivenValueSmall.append(y[row])


        entropyYgivenValueBig = entropy(yGivenValueBig)
        entropyYgivenValueSmall = entropy(yGivenValueSmall)

        entropyYgivenCol = probXSmall * entropyYgivenValueSmall + probXBig * entropyYgivenValueBig
        entropiesLabelGivenColumn[col] = entropyYgivenCol

    #find min of the entropies we get by splitting the different columns
    minValue = min(entropiesLabelGivenColumn.values())
    indexOfMin = list(entropiesLabelGivenColumn.values()).index(minValue)
    #return column with lowest entropy, which means highest information gain
    return indexOfMin 



def chooseSplitColumnWithGini(X, y, thresholdList):
    #number of elements bigger and smaller than threshold in each column
    smallerX, biggerX = countSmallerAndBigger (X, thresholdList)

    #find conditional gini index for each column
    giniLabelGivenColumn = {}

    #(label|value)
    for col in range(X.shape[1]):
        #count values above threshold
        probXBig = biggerX[col]/(smallerX[col]+biggerX[col])
        #count values under threshold
        probXSmall = smallerX[col]/(smallerX[col]+biggerX[col])

        #set of labels given value above and under threshold
        yGivenValueBig = []  
        yGivenValueSmall = [] 

        for row in range(X.shape[0]):
            if(X[row][col] >= thresholdList[col]):
                yGivenValueBig.append(y[row])
            else:
                yGivenValueSmall.append(y[row])

        #count the labels in the set where value is above threshold
        probYBig = probability(yGivenValueBig)
        probYSmall = probability(yGivenValueSmall)  # same but value under

        giniIndexYgivenValueBig = giniIndexLabel(probYBig)
        giniIndexYgivenValueSmall = giniIndexLabel(probYSmall)

        giniYgivenCol = probXSmall*giniIndexYgivenValueSmall+probXBig*giniIndexYgivenValueBig 
        giniLabelGivenColumn[col] = giniYgivenCol

    minValue = min(giniLabelGivenColumn.values())
    indexOfMin = list(giniLabelGivenColumn.values()).index(minValue)
   
    return indexOfMin  


#make two new matrices containing the rows where
#the input column's value would be above/under threshold
def split(X, y, column: int, threshold):
    X1 = np.zeros(shape=(0, X.shape[1]))
    y1 = np.zeros(shape=(0,1))
    X2 = np.zeros(shape=(0, X.shape[1]))
    y2 = np.zeros(shape=(0,1))

    for row in range(X.shape[0]):
        #lower than threshold
        if X[row][column] < threshold: 
            #add row to new matrix
            X1 = np.vstack([X1, X[row, :]]) 
            y1 = np.vstack([y1, y[row]])
            
        else:
            X2 = np.vstack([X2, X[row, :]])  
            y2 = np.vstack([y2, y[row]])
  

    return X1, y1, X2, y2



class Node:
    def __init__(self, col, colName, label, threshold):
        self.left = self
        self.right = self
        self.column = col   
        self.columnName=colName
        self.label=label
        self.threshold=threshold
        self.leaf=False
        self.children=None
        self.majCount=0
        

#find most common label of y
def mostCommonLabel(y):
    if(len(y) == 0):
        return None
    unique, counts = np.unique(y, return_counts = True)
    maxCount = max(counts)
    indexOfMax = np.where(counts == maxCount)
    #return most common label
    return ((unique[indexOfMax])[0])

#check if all labels are the same
def checkLabels(y):
    unique = np.unique(y)
    if(len(unique) == 1):
        return True
    return False

#check if all the values are on the same side of threshold (all values are the same)
def checkValues(X, thresholdList):
    smaller, bigger = countSmallerAndBigger(X, thresholdList)
    if(len(smaller) == 0 or len(bigger) == 0):
        return True
    return False

#return a string of the tree that can be printed
def print_tree(tree: Node, level=0, condition=None):
    ret = "\t" * level + str(tree.columnName) +" (" + str(tree.threshold) + ")" +"label: "+str(tree.label)+" "
    if condition:
        ret += str(condition)
    ret += "\n"

    conditions = ["smaller", "greater"]
    if(tree.left!=tree):
        ret += print_tree(tree.left, level=level + 1, condition=conditions[0])
    if(tree.right!=tree):
        ret += print_tree(tree.right, level=level + 1, condition=conditions[1])
    
    return ret


def ID3(X, y, columnNames, impurity_measure="entropy"):

    #make thresholds for all columns in X
    thresholdList = make_thresholdsX(X)
    
    #check if all labels are the same
    if(checkLabels(y)):
        #return leaf with label
        tree = Node(-1, "Leaf", y[0], None)
        tree.leaf =True
        return tree
    
    #if all values are the same
    elif checkValues(X,thresholdList):
        label = mostCommonLabel(y)
        #return leaf with most common label
        tree = Node(-1, "Leaf", label, None)  
        tree.leaf = True
        return tree
        
    else:
        #find which column to split on
        if(impurity_measure == "gini"):
            column = chooseSplitColumnWithGini(X, y, thresholdList)  #gini index
        else:
            column = chooseSplitColumnWithEntropy(X, y, thresholdList)   #entropy 
        #make node for the column we are splitting on 
        tree = Node(column, columnNames[column], None, thresholdList[column])
    
        #split input matrix and label vector into new matrices and label vectors and do ID3 on them
        X0, y0, X1, y1  = split(X, y, column, thresholdList[column])
        tree.left = ID3(X0, y0, columnNames, impurity_measure)
        tree.right = ID3(X1, y1, columnNames, impurity_measure)
        tree.children=[tree.left, tree.right]
        
        return tree




def predict(x,tree: Node):
    while(tree.leaf==False):
        index=tree.column
        #walk down the tree following the values of the input
        if(x[index]<tree.threshold):
            tree = tree.left
        else:
            tree = tree.right
         
    return tree.label



def main():

    #SKLEARN DECISION TREE FOR COMPARISON:
    skTree= DecisionTreeClassifier()
    skTree=skTree.fit(X_train,y_train)

    pred=skTree.predict(X_test)
    accuracyLearning = accuracy_score(y_test,pred)

    print("accuracy sklearn decision tree:", accuracyLearning)


    #HOMEMADE DECISION TREE:

    #with entropy
    tree1 = ID3(X_train, y_train, columnNamesStart, impurity_measure="entropy")

    #with gini index
    tree2 = ID3(X_train, y_train, columnNamesStart, 
            impurity_measure="gini")



    #if you want to print the trees, just uncomment here
    #it is best to write them to a file, since they are very big

    #print(print_tree(tree1))
    #print(print_tree(tree2))

    prediction1= np.zeros(shape=len(y_test))
    prediction2= np.zeros(shape=len(y_test))


    for i in range(X_test.shape[0]):
        label1 = predict(X_test[i], tree1)
        prediction1[i] = label1

        label2 = predict(X_test[i], tree2)
        prediction2[i] = label2


    accuracyLearning1 = accuracy_score(y_test,prediction1)
    accuracyLearning2 = accuracy_score(y_test,prediction2)

    print("homemade decision tree, entropy:",accuracyLearning1)
    print("homemade decision tree, gini index", accuracyLearning2)
    


main()