import numpy as np
import pandas as pd
from pandas import read_csv

#data prep
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
#data viz
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 

#machine learninig
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
#selection
import sklearn.model_selection
from sklearn.feature_selection import SelectKBest, f_classif

#optuna
import optuna

#misc.
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# Helper functions
def results_to_table(results_dict, tier=3, kind="h_grad", sort=None):
    if tier < 3:
     df=pd.DataFrame.from_dict(results_dict,orient="index")
    else:
     df=pd.DataFrame.from_dict(results_dict,orient="index").stack().to_frame()
        # to break out the lists into columns
     df=pd.DataFrame(df[0].values.tolist(),index=df.index)
    if sort is not None:
     df=df.sort_values(sort, ascending=False)
    if kind=="h_max":
     display(df.style.highlight_max())
    elif kind=="h_grad":
      display(df.style.background_gradient())
    elif kind=="h_quant":
     display(df.style.highlight_quantile(q_left=0.8, q_right=1))
    return df 

    
def modelScorer(classifiers: dict, X_train, y_train, X_test, y_test, scoring='accuracy', subset=None, results_dict=None):
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    resultsScore = []
    if results_dict is None:
        results_dict = {}

    for name in classifiers:
        model_score_dict = {}
        model = classifiers[name]
        model.fit(X_train, y_train)
        kfold = model_selection.KFold(n_splits=10)
        results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        resultsScore.append(results)
        names.append(name)            
        model_score_dict[scoring] = results.mean()
        model_score_dict["std"] = results.std()
        if subset:
         if name in results_dict:
          new_results = {subset : model_score_dict}
        results_dict[name].update(new_results)
        else:
         results_dict[name] = {subset : model_score_dict}
        else:
         results_dict[name] = model_score_dict
        Message = "%s: %f (%f)" % (name, results.mean(), results.std())
        print(Message) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle(f'Algorithm Comparison: {scoring}')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsScore)
    ax.set_xticklabels(names, rotation="vertical")
    ax.set_ylabel(f'Cross-Validation: {scoring} Score')
    plt.show()  
    return results_dict
    classifiers = {
    "Nearest Neighbors" :  KNeighborsClassifier(), 
    "Linear SVM" : SVC(kernel="linear"), 
    "RBF SVM" : SVC(kernel="rbf"), 
    "Gaussian Process" : GaussianProcessClassifier(), 
    "Decision Tree": DecisionTreeClassifier(), 
    "Random Forest" : RandomForestClassifier(), 
    "MLPClassifier" : MLPClassifier(), 
    "AdaBoost" :AdaBoostClassifier(),
    "Naive Bayes" : GaussianNB(), 
    "QuadraticDiscrA" : QuadraticDiscriminantAnalysis(),
    'GradBoostClf': GradientBoostingClassifier()}

    dict_characters = {0: 'Healthy', 1: 'Diabetes'}
    df = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv') # reading in the dataset with pandas
    np.random.seed(42)
    display('info: ----------------------') 
    display(df.info())
    display('Columns: ----------------------',df.columns)
    zero_count_dict = {}
    for col in df.columns:
     if col not in ['Outcome', 'Pregnancies', 'DiabetesPedigreeFunction']:
            #print(col, df[col].value_counts().sort_index().index[0])
            if df[col].value_counts().sort_index().index[0] == 0:
                count = df[col].value_counts().sort_index().iloc[0]
                percent = df[col].value_counts().sort_index().iloc[0]/len(df[col])*100
                list_0 = []
                list_0.append(count)
                list_0.append(percent)
                zero_count_dict[col] = list_0
                
     display(pd.DataFrame.from_dict(zero_count_dict, orient='index', columns=["count", '%']))

     print(df.shape) 
     df = df[df['Glucose'] != 0]

     print(df.shape)
     df = df[df['BMI'] != 0]
     df = df.reset_index().drop("index", axis=1) # to make sure further merging will go as intended
     print(df.shape)

# Luckily, as we can see below, there was no line with only 'Age' as missing value, so we don't lose data due to this action
# in the end we excluded a total of 16 lines (2.08 %)

print('excluded lines:', 752 - 768)
print('in percentage:', "%.2f" % (100 - (752 / 768 * 100)))
df.info()
zero_count_dict = {}
for col in df.columns:
    if col not in ['Outcome', 'Pregnancies', 'DiabetesPedigreeFunction']:
            if df[col].value_counts().sort_index().index[0] == 0:
                count = df[col].value_counts().sort_index().iloc[0]
                percent = df[col].value_counts().sort_index().iloc[0]/len(df[col])*100
                list_0 = []
                list_0.append(count)
                list_0.append(percent)
                zero_count_dict[col] = list_0
display(pd.DataFrame.from_dict(zero_count_dict, orient='index', columns=["count", '%']))
g = sns.heatmap(df.corr(),cmap="Spectral",annot=False)
#see only the 0 vals in SkinThickness for correlation
sns.heatmap(df[df['SkinThickness'] == 0].corr(), cmap="Spectral", annot=False)
df_d0 = df[df['Outcome'] == 0]
df_d1 = df[df['Outcome'] == 1] 
df_d0_samp = df_d0.sample(len(df_d1),replace = False)
df_bal = pd.concat([df_d1, df_d0_samp])
display('balance:',df_bal.Outcome.value_counts())
palette = {0: 'blue', 1: 'red'}

def features_analysis_histplot(df, class_feature):
    """Custom function for making a compact visualization in histplot
        df: the dataframe
        class_feture: the column containing the class variable, this is excluded during the visualization
        returns: none
    """
    plt.figure(figsize = [20, 15]) #setting the figure size for matplotlib
    counter = 0
    print('look at the distribution for all variables')
    for i in df.columns.drop(class_feature):
        counter += 1
        print(counter, ':', i)
        plt.subplot(3, 3, counter)
        sns.histplot(data = df, x = df[str(i)], hue = df[class_feature], multiple  = 'dodge', kde=True, palette=palette)
    plt.plot()
        
features_analysis_histplot(df_bal, 'Outcome')
#Visualizing the outliers with boxplots
pd.plotting.register_matplotlib_converters()

def features_analysis_boxplt(df, class_feature):
    """Custom function for making a compact visualization
        df: the dataframe
        class_feture: the column containing the class variable, this is excluded during the visualization
        returns: none
    """
    plt.figure(figsize = [20, 15])
    counter = 0
    print('look at the distribution for all variables')
    for i in df.columns.drop(class_feature):
        counter += 1
        print(counter, ':', i)
        plt.subplot(3, 3, counter)
        sns.boxplot(x=df[class_feature] ,y = df[str(i)])
    plt.plot()
       
features_analysis_boxplt(df, 'Outcome')
# Splitting the dataset prior manipulation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1), df.Outcome, test_size=0.3, random_state=8)
#Imputing
from sklearn.impute import KNNImputer

X_train[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = X_train[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)

imputer = KNNImputer(missing_values=np.nan)
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train_imp = pd.DataFrame(X_train2)
X_test_imp = pd.DataFrame(X_test2)

#setting the column names back after imputation
X_train_imp.columns = X_train.columns
X_test_imp.columns = X_test.columns
plt.hist(X_train_imp["Insulin"])
plt.hist(X_train_imp["SkinThickness"])
X_train_imp.info()
X_train_imp.isnull().values.any()
X_train = X_train_imp

counter = -1

def objective(trial):

    global counter
    counter = counter + 1
    
    k = trial.suggest_int("k", 1, X_train.shape[1])
    
    skbest = SelectKBest(f_classif, k=k)
    fit_selected_features = skbest.fit(X_train, y_train)
    best_features = skbest.get_feature_names_out()
    
    selected_features = skbest.transform(X_train)
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 10, log=True)

    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    classifier.fit(selected_features, y_train)

    selected_features_test = SelectKBest(f_classif, k=k).fit_transform(X_test, y_test)
    y_pred = classifier.predict(selected_features_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize')
accuracy = study.optimize(objective, n_trials=200)

from sklearn.ensemble import IsolationForest

def find_outliers(df:pd.DataFrame):
    iso = IsolationForest(contamination=0.1)
    outliers = iso.fit_predict(df) #-1 values are Outliers
    print("number of outliers considering the whole dataframe: ",len(outliers[outliers != 1]), f",percent: {len(outliers[outliers != 1])/len(outliers):.2%}")
    outliers_df = pd.DataFrame(outliers)

    print("-----------------")

    outliers_by_cols = {}
    for col in df.columns:
        outliers =iso.fit_predict(df[[col]])
        outliers_by_cols[col] = {}
        outliers_by_cols[col]["Count"] = len(outliers[outliers != 1])
        outliers_by_cols[col]["Percent"] = len(outliers[outliers != 1])/len(outliers)
        print(f"number of outliers in: {col}:",len(outliers[outliers != 1]), f"| percent: {len(outliers[outliers != 1])/len(outliers):.2%}")
    output = pd.DataFrame.from_dict(outliers_by_cols,orient="index")
    output = output.sort_values('Count')
    fig, ax = plt.subplots()
    ax.bar(x=output.index, height=output["Count"])
    ax.bar(x=output.index, height=output["Percent"])
    plt.xticks(rotation=90)
    plt.show()
    return results_to_table(outliers_by_cols, 2)
    find_outliers(X_train_imp)

X_train_imp.describe()
end_results = {} 
end_results = modelScorer(classifiers, X_train_imp, y_train, X_test_imp, y_test, subset="BSL", results_dict=end_results)
results_to_table(end_results, 3)
df.info()
df_prep = df.copy()
df_prep[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = df_prep[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan) 
df_prep = df_prep.drop("Outcome", axis=1)
df_imp_X = imputer.transform(df_prep) 
df_imp_X = pd.DataFrame(df_imp_X)
df_imp_X.columns = df_prep.columns 
df_imp = df_imp_X.copy()
df_imp['Outcome'] = df['Outcome']
df_imp.info()
sns.histplot(x=df_imp['Insulin'], hue=df_imp['Outcome'], kde=True, palette=palette)
plt.show()
sns.histplot(x=df_imp['DiabetesPedigreeFunction'], hue=df_imp['Outcome'], kde=True, palette=palette)
plt.show()
df_imp.info()
df_imp.loc[df_imp['Insulin'].between(0,110,"both"), "insulin_bins"] = 0
df_imp.loc[df_imp['Insulin'].between(110,250,"right"), "insulin_bins"] = 1
df_imp.loc[df_imp['Insulin'].gt(250), "insulin_bins"] = 2
#pedigree - here the cutoff is ~0,5 [>=0.75, 0.75-1.25, 1.25<]
df_imp.loc[df_imp['DiabetesPedigreeFunction'].between(0,0.75,"both"), "pedigree_bins"] = 0
df_imp.loc[df_imp['DiabetesPedigreeFunction'].between(0.75, 1.25,"right"), "pedigree_bins"] = 1
df_imp.loc[df_imp['DiabetesPedigreeFunction'].gt(1.25), "pedigree_bins"] = 2

df_imp.head()
df_imp_d = df_imp.drop(['Insulin','DiabetesPedigreeFunction'], axis=1)
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(df_imp_d.drop('Outcome',axis=1), df_imp_d.Outcome, test_size=0.3, random_state=8)
counter = -1

feature_accuracy = {}

def objective(trial):

    global counter
    global feature_accuracy

    counter = counter + 1
    X_train = X_train_imp
    y_train = y_train_imp
    
    X_test = X_test_imp
    y_test = y_test_imp
    
    k = trial.suggest_int("k", 1, X_train.shape[1])
    
    skbest = SelectKBest(f_classif, k=k)
    fit_selected_features = skbest.fit(X_train, y_train)
    best_features = skbest.get_feature_names_out()
    selected_features = skbest.transform(X_train)

    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, X_train.shape[1], log=True)

    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    classifier.fit(selected_features, y_train)

    selected_features_test = SelectKBest(f_classif, k=k).fit_transform(X_test, y_test)
    y_pred = classifier.predict(selected_features_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    sub_data = {"score" : accuracy, "feat." : best_features}
    feature_accuracy[counter] = sub_data
    return accuracy

study = optuna.create_study(direction='maximize')
accuracy = study.optimize(objective, n_trials=200)
results_to_table(feature_accuracy, tier=2, sort="score")
#preparing dummies from the categorical columns
df_dum = df_imp.drop(['Insulin','DiabetesPedigreeFunction'], axis=1)
df_dummies_ins = pd.get_dummies(df_dum['insulin_bins'])
df_dum = pd.merge(df_dum, df_dummies_ins, how='inner', left_index=True, right_index=True)
df_dum = df_dum.rename(columns={0:'INS_bin_0',1:'INS_bin_1', 2:'INS_bin_2'})

df_dummies_dpf = pd.get_dummies(df_dum['pedigree_bins'])
df_dum = pd.merge(df_dum, df_dummies_dpf, how='inner', left_index=True, right_index=True)
df_dum = df_dum.rename(columns={0:'DPF_bin_0',1:'DPF_bin_1', 2:'DPF_bin_2'})


df_dum = df_dum.drop(['insulin_bins', 'pedigree_bins'], axis=1)

df_dum.head()
df_dum.info()
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df_dum.drop('Outcome',axis=1), df.Outcome, test_size=0.3, random_state=8)

#we add as a new entry of our scores with binning to the dictionary
end_results = modelScorer(classifiers,X_train, y_train, X_test, y_test, subset="BIN", results_dict=end_results)
results_to_table(end_results, 3)
X_train.columns
from sklearn.preprocessing import StandardScaler
sscale = StandardScaler()



SS_X_train = X_train.copy()

sscale.fit(SS_X_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']])
SS_X_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']] = sscale.transform(X = SS_X_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']])
SS_X_train.columns = X_train.columns
SS_X_train.head()
end_results = modelScorer(classifiers, SS_X_train, y_train, X_test, y_test, subset="SSC", results_dict=end_results)
results_to_table(end_results, 3)
# Scale data with MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
mmscale = MinMaxScaler()

mmscale.fit(X_train)

MMS_X_train = X_train.copy()


MMS_X_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']] = mmscale.transform(X = MMS_X_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']])


MMS_X_train.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']
MMS_X_train.head()
end_results = modelScorer(classifiers,MMS_X_train, y_train, X_test, y_test, subset="MMSC", results_dict=end_results)
results_to_table(end_results, 3)
results_to_table(end_results, 3, kind="h_quant")
# preparing the dataset for Optuna
df_dum_X = df_dum.drop("Outcome",axis=1)
df_dum_y = df_dum.Outcome

df_dum_prep = df_dum.copy()

df_dum_prep[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']] = sscale.transform(X = df_dum_prep[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
       'Age', 'INS_bin_0', 'INS_bin_1', 'INS_bin_2', 'DPF_bin_0', 'DPF_bin_1',
       'DPF_bin_2']])

df_dum_prep.head()
MMS_X_train
X_train = MMS_X_train

counter = -1

feature_accuracy = {}

   
def objective(trial):
    global counter
    global feature_accuracy
    
    k = trial.suggest_int("k", 1, X_train.shape[1])
    
    skbest = SelectKBest(f_classif, k=k)
    fit_selected_features = skbest.fit(X_train, y_train)
    best_features = skbest.get_feature_names_out()
    
    selected_features = skbest.transform(X_train)

    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 10, log=True)
    
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    classifier.fit(selected_features, y_train)

    selected_features_test = SelectKBest(f_classif, k=k).fit_transform(X_test, y_test)
    y_pred = classifier.predict(selected_features_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize')
accuracy = study.optimize(objective, n_trials=100)
def objective_2(trial):
    
    x = df_dum_prep.drop("Outcome", axis=1)
    y = df_dum_prep.Outcome

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "NaiveBayes", "MLP"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 0.001, 100, log=True)
        svc_kernel = trial.suggest_categorical("kernel", ["linear","rbf", "poly"])
        classifier_obj = sklearn.svm.SVC(kernel=svc_kernel, C=svc_c, gamma="auto")
    elif classifier_name == "MLP":
        mlp_act = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
        mlp_solver = trial.suggest_categorical("solver", ["lbfgs", "adam"]) # "sgd" solver was removed, due it caused the script to fail
        classifier_obj = MLPClassifier(activation=mlp_act,solver= mlp_solver)
    else:
        nb_var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 0.1, log=True)
        classifier_obj = sklearn.naive_bayes.GaussianNB(
            var_smoothing=nb_var_smoothing
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective_2, n_trials=100)
print(study.best_trial)
from sklearn.calibration import CalibratedClassifierCV

LIN_SVM = SVC(kernel="linear", C=51.68949845761934, probability=True)
clf = CalibratedClassifierCV(LIN_SVM)

clf.fit(MMS_X_train, y_train)
scores = model_selection.cross_val_score(clf, MMS_X_train, y_train, cv=10, scoring="accuracy")
np.mean(scores)
proba = clf.predict_proba(MMS_X_train)
proba
proba_m_df = pd.DataFrame(proba)
proba_m_df.columns = ['Probability_0', 'Probability_1']

proba_df = X_train.merge(proba_m_df, left_index=True, right_index=True).sort_values("Probability_1", ascending=False,)
proba_df.head()
df_dum.info()
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective_(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy
df_dum_prep.describe()