import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from imblearn.over_sampling import SMOTE,SVMSMOTE, RandomOverSampler, ADASYN
from imblearn.pipeline import Pipeline

import pickle

import argparse
import os

class DataModeler():
    def __init__(self):
        self.X = []
        self.y = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.le = None  # Label Encoder
        self.models = {}

    def preprocess(self, X, y):
        """
        Preprocessing the X and y variables

        :param X: Input variable(s)
        :param y: Target variable
        :return:  The preprocessed X and y suitable for modelfinder() method
        """
        if X is None:
            print("Error: X cannot be None or empty. y can be None for predicting the X by the previsouly developed model.")
            return None

        # Remove punctuation
        X = X.str.replace(r'[^\w\d\s]', ' ')

        # Replace whitespace between terms with a single space
        X = X.str.replace(r'\s+', ' ')

        # Remove leading and trailing whitespace
        X = X.str.replace(r'^\s+|\s+?$', '')

        X = X.str.lower()

        # Language name encoder. y can be None for new data without the known language
        if y is not None:
            self.le = LabelEncoder()
            self.le.fit(y)
            y_encoded = self.le.transform(y)


        self.X = X
        self.y = y_encoded

    def logger(self, *args, verbose=1):
        if verbose:
            print(*args)

    def modelfinder(self, splitsize=0.3, verbose=1):
        """
        Finds the best model using f1 score.
        :param splitsize: test set split, default is 0.3
        :param verbose: 0 for hiding and 1 for showing the printouts and reports
        :return: return the best model found
        """
        # Split dataset into train and test sets, using stratify to have same distribution for train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=splitsize, random_state=2,
                                                            stratify=self.y)

        # dictionary containing different classifiers
        classifiers = {'dt': DecisionTreeClassifier(random_state=2, max_depth=100),
                       'lr': LogisticRegression(random_state=2)
                       }

        # dictionary containing different oversamplers
        ovsamplers = {'smote': SMOTE(random_state=2),
                      'random': RandomOverSampler(random_state=2),
                      'adasyn': ADASYN(random_state=2)
                      }

        history = {}
        models = {}
        for clfname, clf in classifiers.items():
            for ovsname, ovs in ovsamplers.items():
                model = Pipeline([('tfidf', TfidfVectorizer()),  # TF-IDF vectorizer (transformer)
                                  ('ovs', ovs),  # oversampler
                                  ('clf', clf)  # classifier
                                  ])

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                history[f"{clfname}_{ovsname}"] = f1_score(y_test, y_test_pred, average='macro')
                models[f"{clfname}_{ovsname}"] = model

                self.logger(clfname, ovsname, verbose=verbose)
                self.logger('', verbose=verbose)
                self.logger(
                    classification_report(self.le.inverse_transform(y_test), self.le.inverse_transform(y_test_pred)),
                    verbose=verbose)
                self.logger("============", verbose=verbose)

        self.logger("Prediction power (f1 score):", verbose=verbose)
        bestmodel = None
        for key, value in sorted(history.items(), key=lambda item: item[1], reverse=True):
            if bestmodel is None:
                bestmodel = models[key]
            self.logger(f"{key}: {value:.2f}", verbose=verbose)

        self.bestmodel = bestmodel
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def extractmodel(self, key=None):
        if key is None:
            return self.models
        else:
            return self.models.get(key)  # get() method avoids raising an exception if the key does not exist

    def plot_confusion_matrix(self,y_true, y_pred, y_lables=None):
        """
        Plots the confusion matrix
        :param y_true: The original y
        :param y_pred: The predicted y
        :param y_lables: The categorical label of y
        :return:
        """
        confmat = confusion_matrix(y_true, y_pred)

        fig = plt.figure()
        ax = plt.subplot()
        sns.heatmap(confmat, annot=True, ax=ax, cmap='Blues', fmt='d');  # annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix');
        if y_lables is not None:
            ax.xaxis.set_ticklabels(y_lables);
            ax.yaxis.set_ticklabels(y_lables);

        ax.axhline(y=0, color='k', linewidth=2)
        ax.axhline(y=confmat.shape[1], color='k', linewidth=2)
        ax.axvline(x=0, color='k', linewidth=2)
        ax.axvline(x=confmat.shape[1], color='k', linewidth=2)


if __name__ == '__main__':

    # get the script directory
    curdir = os.path.dirname(os.path.abspath(__file__))

    # Construct the argument parser
    parser = argparse.ArgumentParser()

    # Switches
    parser.add_argument("-i", dest='inputfilename', required=False, help="Input file name")
    parser.add_argument("-train", required=False, dest='train', action='store_const', const=True,
                        default=False, help="Train the data and save the model")

    # Parse the commandline
    args = parser.parse_args()

    # run in train or test mode
    #mode=args.mode.lower()

    # get the csv filename
    if args.inputfilename is None:
        inputfilename = 'lang_data.csv'
    else:
        inputfilename = args.inputfilename

    # add script path to the data file name, assuming it is next to the python script
    if inputfilename.find("\\")==-1:
        inputfilename = f"{curdir}\\{inputfilename}"

    modelfilename = f"{curdir}\\Lang_classifier_model.pkl"



    # reading the dataset
    df=pd.read_csv(inputfilename)

    # running in train mode
    if args.train:

        # calculate the minumum length of words in each text
        df['min_word_length']=df['text'].apply(lambda x: min([len(s) for s in str(x).split()]))

        # show the plots
        fig, ax =plt.subplots(1,2)
        sns.countplot(data=df,x='language',ax=ax[0])
        sns.countplot(data=df,x='min_word_length',ax=ax[1])


        # remove records with empty (null) text
        df.dropna(inplace=True)

        # initialize X and y variables
        X=df['text']
        y=df['language']

        # initialize the class
        modeler = DataModeler()

        # preprocess the data
        modeler.preprocess(X, y)

        # find the best model
        modeler.modelfinder(verbose=1)

        X_test = modeler.X_test
        y_test=modeler.y_test
        y_test_pred=modeler.bestmodel.predict(X_test)

        print("\n===============\n The classification report for the best model:")
        modeler.logger(classification_report(modeler.le.inverse_transform(y_test), modeler.le.inverse_transform(y_test_pred)),
                        verbose=1)

        # plot the confusion matrix for the test set
        modeler.plot_confusion_matrix(y_test, y_test_pred, modeler.le.classes_)

        # saving the best model

        pickle.dump(modeler, open(modelfilename, "wb"))

        plt.show(block=True)
    else:
        # remove records with empty (null) text
        df.dropna(inplace=True)

        # initialize X and y variables
        X = df['text']
        try:
            y = df['language']
        except:
            y=None

        # load the saved model
        modeler = pickle.load(open(modelfilename, "rb"))

        # preprocess the data
        modeler.preprocess(X, y)

        # save the preprocessed X
        X = modeler.X

        # predict the label for X
        y_pred = modeler.bestmodel.predict(X)

        # perform if the input data has labels
        if y is not None:
            print("\n===============\n The classification report for the best model:")

            y = modeler.y
            modeler.logger(
                classification_report(modeler.le.inverse_transform(y), modeler.le.inverse_transform(y_pred)),
                verbose=1)

            # plot the confusion matrix for the test set
            modeler.plot_confusion_matrix(y, y_pred, modeler.le.classes_)

        ## Save the predicted labels in a new CSV file
        # add the prediction to the loaded data
        df['predicted_language'] = modeler.le.inverse_transform(y_pred)

        # save the prediction result in output.csv
        df.to_csv(f"{curdir}\\output.csv", index=False)

        print('\nPrediction results was saved in "output.csv".')