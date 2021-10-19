import numpy as np
from numpy import sqrt
from numpy import argmax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import streamlit.components.v1 as components
from jinja2 import Template


def import_model():
    # model
    #data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\Project_7\app\data/'
    model = joblib.load('app/data/lgb_light.pkl')

    return model

def import_train_df():
    #data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\Project_7\simplified_app\data/'
    train_df = pd.read_json('app/data/train_df_light.json.gz', orient = 'index')

    return train_df

def import_test_df():
    #data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\Project_7\simplified_app\data/'
    test_df = pd.read_json('app/data/test_df_light.json.gz', orient = 'index')

    return  test_df 

def import_feature_information_df():
     #  data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\Project_7\simplified_app\data/'
    feature_information_df = pd.read_json('app/data/featured_explained.json.gz', orient = 'index')
    feature_information_df.columns = feature_information_df.columns.str.replace('Table', 'Data Source')

    return  feature_information_df

def worst_score_list(df, model):
    df['proba'] = model.predict_proba(df[model.feature_name_], num_iteration=model.best_iteration_)[:, 1]
    return df['proba'].sort_values(ascending=False).head(5).index.to_list()


def initial_cut_off_suggestion (train_df, model):
    train_proba = model.predict_proba(train_df[model.feature_name_], num_iteration=model.best_iteration_)[:, 1]
    #train_df['proba'] = train_proba
    fpr, tpr, thresholds = roc_curve(train_df['TARGET'], train_proba)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    suggested_initial_cut_off = thresholds[ix]
    return suggested_initial_cut_off

def find_optimal_proba (df, model, suggested_initial_cut_off, avg_reimbursed_b4_default , avg_interest):
    
    df['proba'] = model.predict_proba(df[model.feature_name_], num_iteration=model.best_iteration_)[:, 1]

    temp_proba_list = []
    for i in range(0,100,3):
        temp_proba_list.append(i/100)
    temp_proba_list.append(suggested_initial_cut_off)


    optimal_proba_result = 0
    plot_proba = {}

    df['potential_gain'] = avg_interest * df['AMT_CREDIT']
    df['potential_loss'] = (1- avg_reimbursed_b4_default) * df['AMT_CREDIT']
    
    for proba in sorted(temp_proba_list):
        df['result'] = np.nan   
        df['y_proba2'] = np.where(df['proba']>= (proba), 1 , 0)

        df['result'] = np.where((df['TARGET']==0) & (df['y_proba2']==0), df['potential_gain'], df['result'])
        df['result'] = np.where((df['TARGET']==0) & (df['y_proba2']==1), 0, df['result'])
        df['result'] = np.where((df['TARGET']==1) & (df['y_proba2']==0), -df['potential_loss'], df['result'])
        df['result'] = np.where((df['TARGET']==1) & (df['y_proba2']==1), 0, df['result'])


        avg_gains = df['result'].sum() / len(df)
        plot_proba[proba] = avg_gains

        if avg_gains > optimal_proba_result:
            optimal_proba_result = avg_gains
            optimal_proba = proba
            
    return optimal_proba, plot_proba


def print_return_graph(title, opt_perc, plot_proba, suggested_initial_cut_off, column):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()    
    plt.title(title + ' Optimal probability cut off')    
    ax = sns.lineplot(x= plot_proba.keys(), y =plot_proba.values())
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)  
    
    min_key = min(plot_proba, key=plot_proba.get)

    
    # plot square on initial suggested cut off
    plt.vlines(x = suggested_initial_cut_off, ymin=[plot_proba[min_key]], ymax=plot_proba[suggested_initial_cut_off],  linestyle='dotted', color = 'red', label = 'G-Mean threshold' )
    plt.hlines(y = plot_proba[suggested_initial_cut_off], xmin=[0], xmax=suggested_initial_cut_off, linestyle='dotted', color = 'red')
    #plt.text(suggested_initial_cut_off+.01, plot_proba[suggested_initial_cut_off] - plot_proba[suggested_initial_cut_off]*.1, 'Initial cutoff suggestion', horizontalalignment='left', size='medium', color='black')
    
    # plot square on optimal suggested cut off
    plt.vlines(x = opt_perc, ymin=[plot_proba[min_key]], ymax=plot_proba[opt_perc],  linestyle='dashed', color = 'green', label= 'Optimal cut-off ' )
    plt.hlines(y = plot_proba[opt_perc], xmin=[0], xmax=opt_perc, linestyle='dashed', color = 'green')
    #plt.text(opt_perc+.01, plot_proba[opt_perc] + plot_proba[opt_perc]*.1, , horizontalalignment='left', size='medium', color='black')
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("Model's probability prediction cut off")
    plt.ylabel("Average revenue per customer")
    return column.pyplot(fig)

def print_confusion_matrix(df, model, cut_off, title, column):
    df['proba'] = model.predict_proba(df[model.feature_name_], num_iteration=model.best_iteration_)[:, 1]
    df['y_proba2'] = np.where(df['proba']>= (cut_off), 1 , 0)
    cf_matrix = confusion_matrix(df['y_proba2'], df['TARGET'])
    fig = plt.figure()
    group_names = ['True Repaid','False Repaid','False Defaulted','True Defaulted']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names, group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    categories = ['Repaid', 'Defaulted'] 
    sns.heatmap(cf_matrix, annot=labels, xticklabels=categories, yticklabels=categories, fmt='', cmap='summer', cbar=False)
    plt.title(title)
    plt.xlabel('True Labels')
    plt.ylabel('Model Prediction')
    return column.pyplot(fig)

def print_roc_curve(df, model, title, column):
    df['proba'] = model.predict_proba(df.drop(columns = ['TARGET',]), num_iteration=model.best_iteration_)[:, 1]
    fpr, tpr, threshold = roc_curve(df['TARGET'], df['proba'])
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black',sizes=(100,100), label='Validation Threshold suggestion = '\
        +"{0:.0%}".format(threshold[ix]))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Defaulted')
    plt.xlabel('False Defaulted')
    column.pyplot(fig)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



def get_feature_importance(model, shap_values, number_of_features):
    # the -1 index is the customer id
    vals= shap_values[1][-1]
    feature_importance = pd.DataFrame(list(zip(model.feature_name_,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance['feature_importance_vals'] = np.abs(feature_importance['feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    return feature_importance['col_name'].head(number_of_features).to_list()
    
def _build_metric(label, value):
    html_text = """
    <style>
    .metric {
       font-family: "IBM Plex Sans", sans-serif;
       text-align: center;
    }
    .metric .value {
       font-size: 48px;
       line-height: 1.6;
    }
    .metric .label {
       letter-spacing: 2px;
       font-size: 14px;
       text-transform: uppercase;
    }

    </style>
    <div class="metric">
       <div class="value">
          {{ value }}
       </div>
       <div class="label">
          {{ label }}
       </div>
    </div>
    """
    html = Template(html_text)
    return html.render(label=label, value=value)

def metric_row(data):
    columns = st.columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value))

def metric(label, value):
    components.html(_build_metric(label, value))







