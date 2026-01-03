import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_iris_data():
    iris=load_iris()
    print("\n"+"="*50)
    print("IRIS FLOWER CLASSIFICATION")
    print("="*50)
    print(f"\nLoaded {len(iris.data)} flower samples")
    print(f"Species: Setosa, Versicolor, Virginica")
    print(f"Features: {len(iris.feature_names)}")
    return iris

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return X_train,X_test,y_train,y_test

def knn_model(X_train,X_test,y_train,y_test):
    print("\n"+"-"*50)
    print("K-Nearest Neighbors (KNN)")
    print("-"*50)
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    acc=accuracy_score(y_test,predictions)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test,predictions))
    return model,acc

def tree_model(X_train,X_test,y_train,y_test):
    print("\n"+"-"*50)
    print("Decision Tree")
    print("-"*50)
    model=DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    acc=accuracy_score(y_test,predictions)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test,predictions))
    return model,acc

def test_prediction(model,species_names):
    print("\n"+"="*50)
    print("Testing Prediction")
    print("="*50)
    sample=[[5.1,3.5,1.4,0.2]]
    result=model.predict(sample)
    print(f"\nFlower measurements: {sample[0]}")
    print(f"Predicted: {species_names[result[0]]}")

if __name__=="__main__":
    iris=load_iris_data()
    X_train,X_test,y_train,y_test=split_data(iris.data,iris.target)
    knn,knn_accuracy=knn_model(X_train,X_test,y_train,y_test)
    tree,tree_accuracy=tree_model(X_train,X_test,y_train,y_test)
    print("\n"+"="*50)
    print("Comparison")
    print("="*50)
    print(f"KNN: {knn_accuracy*100:.2f}%")
    print(f"Decision Tree: {tree_accuracy*100:.2f}%")
    
    if knn_accuracy>tree_accuracy:
        print("\nKNN performed better!")
        best=knn
    else:
        print("\nDecision Tree performed better!")
        best=tree

    test_prediction(best,iris.target_names)
    
    print("\nDone!")

