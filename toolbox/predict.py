import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from random import sample
import pickle
import joblib
import matplotlib.ticker as mtick
from lightgbm import LGBMClassifier
from sklearn import metrics
from random import sample

data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\projet 7\pickled/'

def import_df_and_model():
    # import df and model
    data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\projet 7\pickled/'
    model = joblib.load(data_directory + 'lgb.pkl')
    train_df = pd.read_pickle(data_directory + "train_df.pkl")
    test_df = pd.read_pickle((data_directory + "test_df.pkl"))

    return model, train_df, test_df

def predict_single_proba (id_customer):
    model, train_df, test_df = import_df_and_model()
    
    proba = model.predict_proba(test_df.drop(columns = ['TARGET',]).loc[[int(id_customer)]], num_iteration = model.best_iteration_)[0][0]
    
    return proba

# instead of using 0.5, we propose a different cutoff based on training data set
def initial_cut_off_suggestion (train_df, model):
    train_proba = model.predict_proba(train_df.drop(columns = ['TARGET',]), num_iteration=model.best_iteration_)[:, 0]
    train_df['proba'] = train_proba

    fpr, tpr, threshold = metrics.roc_curve(train_df['TARGET'], train_proba, )
    suggested_initial_cut_off = metrics.auc(fpr, tpr)
    return suggested_initial_cut_off

# ask user to select a avg loss for default customers and gains for good ones
def find_optimal_proba (test_df ,train_df, suggested_initial_cut_off):
    print('Customer class information')
    print('Average credit: ', int(test_df.AMT_CREDIT.mean()))
    print('Maximum credit: ', int(test_df.AMT_CREDIT.max()))
    print('Minimum credit: ', int(test_df.AMT_CREDIT.min()))
    print()

    print('Average credit percentage reimbursed before default? Input a number between 0 and 1')
    avg_reimbursed_b4_default = float(input())
    print('Enter customer Interest rate or average of customer group above. Input a number between 0 and 1')
    avg_interest = float(input())

    temp_proba_list = []
    for i in range(0,100,3):
        temp_proba_list.append(i/100)
    temp_proba_list.append(suggested_initial_cut_off)


    optimal_proba_result = 0
    plot_proba_dic = {}

    train_df['potential_gain'] = avg_interest * train_df['AMT_CREDIT']
    train_df['potential_loss'] = avg_reimbursed_b4_default * train_df['AMT_CREDIT']

    for proba in sorted(temp_proba_list):
        train_df['result'] = np.nan   
        train_df['y_proba2'] = np.where(train_df['proba']>= (proba), 1 , 0)

        train_df['result'] = np.where((train_df['TARGET']==0) & (train_df['y_proba2']==0), train_df['potential_gain'], train_df['result'])
        train_df['result'] = np.where((train_df['TARGET']==0) & (train_df['y_proba2']==1), -train_df['potential_gain'], train_df['result'])
        train_df['result'] = np.where((train_df['TARGET']==1) & (train_df['y_proba2']==0), -train_df['potential_loss'], train_df['result'])
        train_df['result'] = np.where((train_df['TARGET']==1) & (train_df['y_proba2']==1), train_df['potential_loss'], train_df['result'])


        avg_gains = train_df['result'].sum() / len(train_df)
        plot_proba_dic[proba] = avg_gains

        if avg_gains > optimal_proba_result:
            optimal_proba_result = avg_gains
            optimal_proba = proba
            
    return optimal_proba, plot_proba_dic

    
# plot returns graph
def print_return_graph(plot_proba_dic, suggested_initial_cut_off):

    opt_perc = max(plot_proba_dic, key=plot_proba_dic.get)
    print('Based on the information imputed here is the graph of returns on the training data')

    plt.figure(figsize=(10,10))
    plt.title('Optimal probability cut off')
    ax = sns.lineplot(x= plot_proba_dic.keys(), y =plot_proba_dic.values())
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)  
    
    # plot square on initial suggested cut off
    plt.vlines(x = suggested_initial_cut_off, ymin=[0], ymax=plot_proba_dic[suggested_initial_cut_off],  linestyle='dotted', color = 'red', label = 'Initial cutoff suggestion' )
    plt.hlines(y = plot_proba_dic[suggested_initial_cut_off], xmin=[0], xmax=suggested_initial_cut_off, linestyle='dotted', color = 'red')
    #plt.text(suggested_initial_cut_off+.01, plot_proba_dic[suggested_initial_cut_off] - plot_proba_dic[suggested_initial_cut_off]*.1, 'Initial cutoff suggestion', horizontalalignment='left', size='medium', color='black')
    
    # plot square on optimal suggested cut off
    plt.vlines(x = opt_perc, ymin=[0], ymax=plot_proba_dic[opt_perc],  linestyle='dashed', color = 'green', label= 'Optimal cutoff ' )
    plt.hlines(y = plot_proba_dic[opt_perc], xmin=[0], xmax=opt_perc, linestyle='dashed', color = 'green')
    #plt.text(opt_perc+.01, plot_proba[opt_perc] + plot_proba[opt_perc]*.1, , horizontalalignment='left', size='medium', color='black')
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("Model's probability prediction")
    plt.ylabel("Average return per customer")
    plt.show()

# Return prediction
def return_prediction(ID):
    ID = int(ID)
    


# 






