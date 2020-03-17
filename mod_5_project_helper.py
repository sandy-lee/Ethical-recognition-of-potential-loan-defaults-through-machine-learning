#!/usr/bin/env python3


def set_environment():
    import pandas as pd
    import warnings

    """
    This is a function to set environment  
    variables in the notebook. Takes no arguments. 
    Returns environment settings.
    """

    return [pd.set_option('display.max_columns', None),
            warnings.filterwarnings('ignore'),
            pd.set_option('display.max_columns', None)]


def drop_column_keyword_search(dataframe = None, keywords = None):
    """
    This function deletes columns from a Pandas dataframe it receives an argument based on keywords also as arguments
    """

    regex = ""
    for word in range(0, len(keywords)):
        regex += keywords[word] + "|"
    regex = regex[:-1]
    updated_df = dataframe[dataframe.columns.drop(list
                                                  (dataframe.filter
                                                  (regex=regex, axis=1)))]
    return updated_df


def data_cleanup(dataframe = None):

    """
    This is a simple procedural function to clean up the dataset based on the EDA process; it returns a cleaned dataset
    and its primary purpose is to keep the Jupyter Notebook cleaner
    """

    import pandas as pd
    import numpy as np

    df = dataframe

    df.drop_duplicates(inplace = True) #drop 10,728 duplicate rows
    df.drop(index = 100000, axis = 0, inplace = True) #delete last blank row in the data set
    df.drop(df[df["Current Loan Amount"] == 99999999].index, inplace = True) #delete 11484 rows where the loan amount is 99999999
    df.dropna(subset = ['Credit Score'], inplace = True) #delete 19154 rows where Credit Score is missing
    df.dropna(subset = ['Years in current job'], inplace = True) #delete 2564 rows where Years in current job is missing 
    df["Years in current job"].replace(to_replace = "year.*", value = "",inplace = True, regex = True) #remove 'years' or 'year' from string
    df.drop(df[df["Home Ownership"] == "HaveMortgage"].index, inplace = True) #delete 120 rows with "HaveMortgage" as meaning isn't clear
    df.dropna(subset = ['Months since last delinquent'], inplace = True) #delete 30,000 rows where data is missing 
    df.drop(columns = "Maximum Open Credit", inplace = True) #delete Maximum Open Credit columns as some of these numbers are huge e.g. 798255370.0
    df = drop_column_keyword_search(df, ["Loan ID", "Customer ID"]) #delete features 'Loan ID' and 'Customer ID' as they don't add anything
    df = df.fillna(df.median()) #fill remaining values with 
    df["Purpose"].replace({   #rename and consolidate categorical variables for purpose
              "Business Loan": "business_loan",
              "Medical Bills": "medical bills",
              "Educational Expenses": "educational_expenses",
              "Buy House": "buy_house",
              "Buy a Car": "buy_a_car",
              "Debt Consolidation": "debt_consolidation",
              "Home Improvements": "home_improvements",
              "Take a Trip": "take_a_trip",
              "vacation": "take_a_trip",
              "Major Purchase": "other",
              "Other": "other", 
              "renewable_energy": "home_improvements",
              "small_business": "business_loan",
              "moving": "home_improvements",
              "major_purchase": "major_purchase",
              "wedding": "wedding"
              }, inplace=True)
    df["Home Ownership"].replace({    #rename categorical variables for Home Ownership 
              "Home Mortgage": "mortgage",
              "Rent": "rent",
              "Own Home": "own_home",
              }, inplace=True)
    df["Term"].replace({    #rename categorical variables for Term
              "Long Term": "long_term",
              "Short Term": "short_term",
              }, inplace=True)
    df["Loan Status"].replace({    #rename categorical variables for Loan Status
              "Fully Paid": "fully_paid",
              "Charged Off": "default",
              }, inplace=True)
    df.columns = ['loan_status',    #rename columns to make the dataset easier to work with using . notation
              'loan_amount',
              'term',
              'credit_score',
              'annual_income',
              'years_in_current_job',
              'home_ownership',
              'loan_purpose',
              'monthly_debt',
              'years_of_credit_history',
              'months_since_last_delinquent',
              'number_of_open_accounts',
              'number_of_credit_problems',
              'current_credit_balance',
              'bankruptcies',
              'tax_liens']
    df.reset_index(inplace = True); #reset index
    df.drop(columns = "index", inplace = True) #remove extra index columns as not needed
    df['loan_status_binary_value'] = np.where(df["loan_status"] == "fully_paid",1,0) #add a binary columns for loan_status with 1 = "fully_paid" and 0 = "default"

    return df

def correlation_matrix(df = None):
    
    """
    This is a function that returns a nicely formatted correlation matrix for a dataframe
    It takes a Pandas dataframe as an argument and returns a correlation object for 
    notebook embedding
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set(style="white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return(heatmap)

def one_hot_encoding(df_train=None, df_val=None):

    """
    This is a procdural function to one-hot-encode our notebook dataframes.
    It takes a Pandas dataframe as an argument, seperates out categorical & 
    continous variables and returns a fully one-hot-encoded dataframe

    To do:

    [] make this into a function that can take any dataframe and return one-hot-encoding
    """

    import pandas as pd

    X_train = df_train
    X_val = df_val

    X_train_categorical = pd.concat([X_train.term,    #make a dataframe of categorical variables - training data
                                 X_train.years_in_current_job, 
                                 X_train.home_ownership, 
                                 X_train.loan_purpose], 
                                 axis = 1) 
    
    X_train_continuous = pd.concat([X_train.loan_amount,    #make a dataframe of continous variables - training data
                                X_train.credit_score, 
                                X_train.annual_income, 
                                X_train.monthly_debt, 
                                X_train.years_of_credit_history,
                                X_train.months_since_last_delinquent,
                                X_train.number_of_open_accounts,
                                X_train.current_credit_balance,
                                X_train.bankruptcies,
                                X_train.tax_liens], 
                                axis = 1)

    X_train_one_hot_encoded = pd.get_dummies(X_train_categorical)    #one hot encode categorical variables - training data
                                                                     #for everything but decision trees we should do drop_first = True
    
    X_train_all_numbers = pd.concat([X_train_continuous, X_train_one_hot_encoded], axis = 1)    #stitch the one hot encoded dataframe back together - training data

    X_val_categorical = pd.concat([X_val.term,    #make a dataframe of categorical variables - validation data
                               X_val.years_in_current_job, 
                               X_val.home_ownership, 
                               X_val.loan_purpose], 
                               axis = 1) 

    X_val_continuous = pd.concat([X_val.loan_amount,    #make a dataframe of continous variables - validation data
                              X_val.credit_score, 
                              X_val.annual_income, 
                              X_val.monthly_debt, 
                              X_val.years_of_credit_history,
                              X_val.months_since_last_delinquent,
                              X_val.number_of_open_accounts,
                              X_val.current_credit_balance,
                              X_val.bankruptcies,
                              X_val.tax_liens], 
                              axis = 1)

    X_val_one_hot_encoded = pd.get_dummies(X_val_categorical)    #one hot encode categorical variables - validation data
                                                                 #for everything but decision trees we should do drop_first = True
    
    X_val_all_numbers = pd.concat([X_val_continuous, X_val_one_hot_encoded], axis = 1)    #stitch the one hot encoded dataframe back together - validation data

    return(X_train_all_numbers, X_val_all_numbers)


def scores(model,X_train,X_val,y_train,y_val):

    """
    This is a function to calulate roc_auc scores for classifier performance comparison
    """
    from sklearn.metrics import roc_auc_score
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))


def roc_plot(model,X_train,y_train,X_val,y_val):

    """
    This is a function to plot a ROC curve for classifier performance checking
    """
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    plt.figure(figsize=(7,7))
    for data in [[y_train, train_prob],[y_val, val_prob]]: # ,[y_test, test_prob]
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()


def annot(fpr,tpr,thr):
    """
    This is a function for annotating FPR and TPR on ROC curve plots
    """
    import matplotlib.pyplot as plt
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1


def opt_plots(opt_model):

    """
    This is a function for plotting heatmaps of hyperparameter performance
    for GridSearchCV optimisation when training classifiers
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')