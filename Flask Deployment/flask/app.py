__author__ = 'kin_chun'
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response , url_for,redirect
import joblib
import os
import io
import base64
import csv
import pandas
import pydotplus
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus 
import warnings
import os.path
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore")

#python to open csv for MAC
folder_MAC = 'C:/Users/Admin/Desktop/CSV file/Backup/Flask Testing/Display'
file_name_MAC = 'Master_MAC_Attack.csv'
file_path_MAC = os.path.join(folder_MAC, file_name_MAC)

#python to open csv for DHCP
folder_DHCP = 'C:/Users/Admin/Desktop/CSV file/Backup/Flask Testing/Display'
file_name_DHCP = 'Master_Spoofing_Attack.csv'
file_path_DHCP = os.path.join(folder_DHCP, file_name_DHCP)

#python to open csv for ICMP
folder_ICMP = 'C:/Users/Admin/Desktop/CSV file/Backup/Flask Testing/Display'
file_name_ICMP = 'Master_ICMP_Attack.csv'
file_path_ICMP = os.path.join(folder_ICMP, file_name_ICMP)

#python to open csv for SYN
folder_SYN = 'C:/Users/Admin/Desktop/CSV file/Backup/Flask Testing/Display'
file_name_SYN = 'Master_SYN_Attack.csv'
file_path_SYN = os.path.join(folder_SYN, file_name_SYN)


app = Flask(__name__)

def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):
            classificationReport = classification_report(Y_validation, newpredictions_rfc) 
            classificationReport = classificationReport.replace('\n\n', '\n')
            classificationReport = classificationReport.replace(' / ', '/')
            lines = classificationReport.split('\n')

            classes, plotMat, support, class_names = [], [], [], []
            for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
                t = line.strip().split()
                if len(t) < 2:
                    continue
                classes.append(t[0])
                v = [float(x) for x in t[1: len(t) - 1]]
                support.append(int(t[-1]))
                class_names.append(t[0])
                plotMat.append(v)

            plotMat = np.array(plotMat)
            xticklabels = ['Precision', 'Recall', 'F1-score']
            yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                           for idx, sup in enumerate(support)]

            plt.imshow(plotMat, interpolation='nearest', cmap='RdBu', aspect='auto')
            plt.title('Classification Report')
            plt.colorbar()
            plt.xticks(np.arange(3), xticklabels, rotation=45)
            plt.yticks(np.arange(len(classes)), yticklabels)

            upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
            lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
            for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
                plt.text(j, i, format(plotMat[i, j], '.2f'),
                         horizontalalignment="center",
                         color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

            plt.ylabel('Metrics')
            plt.xlabel('Classes')
            plt.tight_layout()
            plt.show()

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
        
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        if(len(t)==0):
            break
        else:
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            print(v)
            plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/deploy', methods=['POST'])
def deploy():

    attackType = request.form["attack"]
    modelType = request.form['model']
    ratio = request.form['ratio']
    dataset = request.form['dataset']
        
    if modelType == "lr":
        classifier = LogisticRegression()
    elif modelType == "lda":
        classifier = LinearDiscriminantAnalysis()
    elif modelType == "knn":
        classifier = KNeighborsClassifier()
    elif modelType == "dt":
        classifier = DecisionTreeClassifier()

    if ratio == "ratio70":
        validation_size = 0.30
    elif ratio == "ratio80":
        validation_size = 0.20

    if dataset == "mac_dataset":
        file_path = file_path_MAC
    elif dataset == "dhcp_dataset":
        file_path = file_path_DHCP
    elif dataset == "syn_dataset":
        file_path = file_path_SYN
    elif dataset == "icmp_dataset":
        file_path = file_name_ICMP
    
    # import dataset
    if dataset == "mac_dataset":
        dataset = pandas.read_csv("Master_MAC_Attack.csv")
    elif dataset == "dhcp_dataset":
        dataset = pandas.read_csv("Master_Spoofing_Attack.csv")
    elif dataset == "syn_dataset":
        dataset = pandas.read_csv("Master_SYN_Attack.csv")
    elif dataset == "icmp_dataset":
        dataset = pandas.read_csv("Master_ICMP_Attack.csv")
        
    #(Array has to be defined to avoid UnboundLocalError)
    # split dataset
    array = dataset.values
    print (array)
    if attackType == "mac":
        X = array[:,0:6]
        Y = array[:,6]
    elif attackType == "dhcp":
        X = array[:,0:8]
        Y = array[:,8]
    elif attackType == "syn":
        X = array[:,0:8]
        Y = array[:,8]
    elif attackType == "icmp":
        X = array[:,0:8]
        Y = array[:,8]
 
    # split dataset
    # array = dataset.values
    # print (array)
    # X and Y has already been configure during the attackType
    #X = array[:,0:8]
    #Y = array[:,8]
    # Validation size has been configure during the ratio selection
    #validation_size = 0.30
    seed = 42
    
    # Split dataset into training set and test set
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    X_train_set, X_test, Y_train_set, Y_test = model_selection.train_test_split(X_train, Y_train, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # Fit algorithm model

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))


    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "Highest Accuracy is %s with Accuracy: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    
    #Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    # plt.clf()
    # plt.cla()
    # plt.close()
    
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='png')
    my_stringIObytes.seek(0)
    my_base64_pngData = base64.b64encode(my_stringIObytes.read())
    data_uri = "data:image/png;base64," + my_base64_pngData.decode("ascii")
    
    #plt.show()
    
    if modelType == "lr":
        # Testing For Logistic Regression
            print("-------------------------------------------------------")

        # Make predictions on validation dataset
            print("\n Logistic Regression results on test set \n")
        # Create Logistic Regression classifer object
            logistic = LogisticRegression()
        # Train Logistic Regression Classifer
            logistic.fit(X_train_set, Y_train_set)
        #saving the model using joblib 
            filename = 'logistic_finalized_DT_model.sav'
            joblib.dump(logistic, filename)
        # load the model from disk
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X_test, Y_test)
            print (result)
        #Predict the response for test dataset
            predictions_rfc = logistic.predict(X_test)
            print("\nLogistic Regression accuracy test: \n")
            print(accuracy_score(Y_test, predictions_rfc))
            print(confusion_matrix(Y_test, predictions_rfc))
            print(classification_report(Y_test, predictions_rfc, zero_division=0))
            
            Train_Accuracy = accuracy_score(Y_test, predictions_rfc)
            Train_Confusion_Matrix = confusion_matrix(Y_test, predictions_rfc)
            Train_Classification_Report = classification_report(Y_test, predictions_rfc)
            # matrix = plot_confusion_matrix(logistic, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()



        # Make predictions on test dataset
            print("\nLogistic Regression results on final validation \n")
            newlogistic = DecisionTreeClassifier()
            newlogistic.fit(X_train_set, Y_train_set)
            newpredictions_rfc = newlogistic.predict(X_validation)
            print("\nLogistic Regression accuracy validation: \n")
            print(accuracy_score(Y_validation, newpredictions_rfc))
            print(confusion_matrix(Y_validation, newpredictions_rfc))
            print(classification_report(Y_validation, newpredictions_rfc))
            df = dataset.reset_index(drop = False)
            
            Test_Accuracy = accuracy_score(Y_validation, newpredictions_rfc)
            Test_Confussion_Matrix = confusion_matrix(Y_validation, newpredictions_rfc)
            Test_Classification_Report = classification_report(Y_validation, newpredictions_rfc) 
            matrix = plot_confusion_matrix(newlogistic, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()
            
            # #Compare Algorithms
            # fig = plt.figure()
            # fig.suptitle('Confusion Matrix')
            # ax = fig.add_subplot(111)
            # plt.boxplot(results)
            # ax.set_xticklabels(names)
            
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_matrix = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            plt.close()
            
            
            
            # plot_classification_report(classification_report(Y_validation, newpredictions_rfc) )
            # The following is classification report coding to display visualization (heatmap only)
            
            lines = classification_report(Y_validation, newpredictions_rfc).split('\n')
            
            classes = []
            plotMat = []
                
            for line in lines[2 : (len(lines) - 3)]:
                #print(line)
                t = line.split()
                print(t)
                if(len(t)==0):
                    break
                else:
                    classes.append(t[0])
                    v = [float(x) for x in t[1: len(t) - 1]]
                    print(v)
                    plotMat.append(v)

            # if with_avg_total:
                # aveTotal = lines[len(lines) - 1].split()
                # classes.append('avg/total')
                # vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
                # plotMat.append(vAveTotal)


            plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Classification report ')
            plt.colorbar()
            x_tick_marks = np.arange(3)
            y_tick_marks = np.arange(len(classes))
            plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
            plt.yticks(y_tick_marks, classes)
            plt.tight_layout()
            plt.ylabel('Classes')
            plt.xlabel('Measures') 
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_report = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            
            # plt.close()
            # Classification Report (Not accurate)
            #-----------------------------------------------------------------------------------
            #-----------------------------------------------------------------------------------
            # classificationReport = classification_report(Y_validation, newpredictions_rfc) 
            # plot_classification_report(classificationReport)
            # plt.show()
            
            # Instantiate the classification model and visualizer


    elif modelType == "lda":
        # Testing For Linear Discriminant Analysis
            print("-------------------------------------------------------")
        # Make predictions on validation dataset
            print("\n Linear Discriminant Analysis results on 30% test set \n")
        # Create Logistic Regression classifer object
            lda = LinearDiscriminantAnalysis()
        # Train Linear Discriminant Analysis Classifer
            lda.fit(X_train_set, Y_train_set)
        #saving the model using joblib 
            filename = 'lda_finalized_DT_model.sav'
            joblib.dump(lda, filename)
        # load the model from disk
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X_test, Y_test)
            print (result)
        #Predict the response for test dataset
            predictions_rfc = lda.predict(X_test)
            print("\nLinear Discriminant Analysis accuracy test: \n")
            print(accuracy_score(Y_test, predictions_rfc))
            print(confusion_matrix(Y_test, predictions_rfc))
            print(classification_report(Y_test, predictions_rfc, zero_division=0))
            
            Train_Accuracy = accuracy_score(Y_test, predictions_rfc)
            Train_Confusion_Matrix = confusion_matrix(Y_test, predictions_rfc)
            Train_Classification_Report = classification_report(Y_test, predictions_rfc)
            # matrix = plot_confusion_matrix(lda, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()

        # Make predictions on test dataset
            print("\nLinear Discriminant Analysis results on final 30% validation \n")
            newlda = DecisionTreeClassifier()
            newlda.fit(X_train_set, Y_train_set)
            newpredictions_rfc = newlda.predict(X_validation)
            print("\nLinear Discriminant Analysis accuracy validation: \n")
            print(accuracy_score(Y_validation, newpredictions_rfc))
            print(confusion_matrix(Y_validation, newpredictions_rfc))
            print(classification_report(Y_validation, newpredictions_rfc))
            df = dataset.reset_index(drop = False)
           

            Test_Accuracy = accuracy_score(Y_validation, newpredictions_rfc)
            Test_Confussion_Matrix = confusion_matrix(Y_validation, newpredictions_rfc)
            Test_Classification_Report = classification_report(Y_validation, newpredictions_rfc) 
            matrix = plot_confusion_matrix(newlda, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()
            
            # #Compare Algorithms
            # fig = plt.figure()
            # fig.suptitle('Confusion Matrix')
            # ax = fig.add_subplot(111)
            # plt.boxplot(results)
            # ax.set_xticklabels(names)
            
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_matrix = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            plt.close()
            
            
            
            lines = classification_report(Y_validation, newpredictions_rfc).split('\n')
            
            classes = []
            plotMat = []
                
            for line in lines[2 : (len(lines) - 3)]:
                #print(line)
                t = line.split()
                print(t)
                if(len(t)==0):
                    break
                else:
                    classes.append(t[0])
                    v = [float(x) for x in t[1: len(t) - 1]]
                    print(v)
                    plotMat.append(v)
                    
            plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Classification report ')
            plt.colorbar()
            x_tick_marks = np.arange(3)
            y_tick_marks = np.arange(len(classes))
            plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
            plt.yticks(y_tick_marks, classes)
            plt.tight_layout()
            plt.ylabel('Classes')
            plt.xlabel('Measures') 
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_report = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            

    elif modelType == "knn":
        # Testing For KNeighbors
            print("-------------------------------------------------------")

        # Make predictions on validation dataset
            print("\n KNeighbors results on 30% test set \n")
        # Create K Neighbors classifer object
            KNeighbors = KNeighborsClassifier()
        # Train KNeighbors Classifer
            KNeighbors.fit(X_train_set, Y_train_set)
        #saving the model using joblib 
            filename = 'kneighbors_finalized_DT_model.sav'
            joblib.dump(KNeighbors, filename)
        # load the model from disk
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X_test, Y_test)
            print (result)
        #Predict the response for test dataset
            predictions_rfc = KNeighbors.predict(X_test)
            print("\nK Neighbors accuracy test: \n")
            print(accuracy_score(Y_test, predictions_rfc))
            print(confusion_matrix(Y_test, predictions_rfc))
            print(classification_report(Y_test, predictions_rfc, zero_division=0))
            
            Train_Accuracy = accuracy_score(Y_test, predictions_rfc)
            Train_Confusion_Matrix = confusion_matrix(Y_test, predictions_rfc)
            Train_Classification_Report = classification_report(Y_test, predictions_rfc)
            # matrix = plot_confusion_matrix(KNeighbors, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()

        # Make predictions on test dataset
            print("\nK Neighbors results on final 30% validation \n")
            newKNeighbors = DecisionTreeClassifier()
            newKNeighbors.fit(X_train_set, Y_train_set)
            newpredictions_rfc = newKNeighbors.predict(X_validation)
            print("\nK Neighbors accuracy validation: \n")
            print(accuracy_score(Y_validation, newpredictions_rfc))
            print(confusion_matrix(Y_validation, newpredictions_rfc))
            print(classification_report(Y_validation, newpredictions_rfc))
            df = dataset.reset_index(drop = False)
            
            Test_Accuracy = accuracy_score(Y_validation, newpredictions_rfc)
            Test_Confussion_Matrix = confusion_matrix(Y_validation, newpredictions_rfc)
            Test_Classification_Report = classification_report(Y_validation, newpredictions_rfc) 
            matrix = plot_confusion_matrix(newKNeighbors, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()
            
            # #Compare Algorithms
            # fig = plt.figure()
            # fig.suptitle('Confusion Matrix')
            # ax = fig.add_subplot(111)
            # plt.boxplot(results)
            # ax.set_xticklabels(names)
            
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_matrix = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            plt.close()
            
            
            lines = classification_report(Y_validation, newpredictions_rfc).split('\n')
            
            classes = []
            plotMat = []
                
            for line in lines[2 : (len(lines) - 3)]:
                #print(line)
                t = line.split()
                print(t)
                if(len(t)==0):
                    break
                else:
                    classes.append(t[0])
                    v = [float(x) for x in t[1: len(t) - 1]]
                    print(v)
                    plotMat.append(v)
            
            plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Classification report ')
            plt.colorbar()
            x_tick_marks = np.arange(3)
            y_tick_marks = np.arange(len(classes))
            plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
            plt.yticks(y_tick_marks, classes)
            plt.tight_layout()
            plt.ylabel('Classes')
            plt.xlabel('Measures') 
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_report = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            
    elif modelType == "dt":
        # Make predictions on validation dataset
            print("\n CART results on 30% test set \n")
        # Create Decision Tree classifer object
            cart = DecisionTreeClassifier()
        # Train Decision Tree Classifer
            cart.fit(X_train_set, Y_train_set)
        #saving the model using joblib 
            filename = 'decision_finalized_DT_model.sav'
            joblib.dump(cart, filename)
        # load the model from disk
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X_test, Y_test)
            print (result)
        #Predict the response for test dataset
            predictions_rfc = cart.predict(X_test)
            print("\nCART accuracy test: \n")
            print(accuracy_score(Y_test, predictions_rfc))
            print(confusion_matrix(Y_test, predictions_rfc))
            print(classification_report(Y_test, predictions_rfc))
            
            Train_Accuracy = accuracy_score(Y_test, predictions_rfc)
            Train_Confusion_Matrix = confusion_matrix(Y_test, predictions_rfc)
            Train_Classification_Report = classification_report(Y_test, predictions_rfc)
            # matrix = plot_confusion_matrix(cart, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()
            
        # Make predictions on test dataset
            print("\nCART results on final 30% validation \n")
            newcart = DecisionTreeClassifier()
            newcart.fit(X_train_set, Y_train_set)
            newpredictions_rfc = newcart.predict(X_validation)
            print("\nCART accuracy validation: \n")
            print(accuracy_score(Y_validation, newpredictions_rfc))
            print(confusion_matrix(Y_validation, newpredictions_rfc))
            print(classification_report(Y_validation, newpredictions_rfc))
            df = dataset.reset_index(drop = False)
            
            Test_Accuracy = accuracy_score(Y_validation, newpredictions_rfc)
            Test_Confussion_Matrix = confusion_matrix(Y_validation, newpredictions_rfc)
            Test_Classification_Report = classification_report(Y_validation, newpredictions_rfc) 
            matrix = plot_confusion_matrix(newcart, X_test, Y_test, cmap=plt.cm.Blues)
            # plt.show()
            
            # #Compare Algorithms
            # fig = plt.figure()
            # fig.suptitle('Confusion Matrix')
            # ax = fig.add_subplot(111)
            # plt.boxplot(results)
            # ax.set_xticklabels(names)
            
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_matrix = "data:image/png;base64," + my_base64_pngData.decode("ascii")
            plt.close()
            
            lines = classification_report(Y_validation, newpredictions_rfc).split('\n')
            
            classes = []
            plotMat = []
                
            for line in lines[2 : (len(lines) - 3)]:
                #print(line)
                t = line.split()
                print(t)
                if(len(t)==0):
                    break
                else:
                    classes.append(t[0])
                    v = [float(x) for x in t[1: len(t) - 1]]
                    print(v)
                    plotMat.append(v)
            plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Classification report ')
            plt.colorbar()
            x_tick_marks = np.arange(3)
            y_tick_marks = np.arange(len(classes))
            plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
            plt.yticks(y_tick_marks, classes)
            plt.tight_layout()
            plt.ylabel('Classes')
            plt.xlabel('Measures') 
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='png')
            my_stringIObytes.seek(0)
            my_base64_pngData = base64.b64encode(my_stringIObytes.read())
            data_uri_report = "data:image/png;base64," + my_base64_pngData.decode("ascii")
        
    return render_template('deploy.html', data=msg, data2=Train_Accuracy, data3=data_uri_matrix, data4=Test_Classification_Report, comparison_uri=data_uri, data5=Test_Accuracy, data6=Test_Confussion_Matrix, data7=Test_Classification_Report, data8=data_uri_report)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
