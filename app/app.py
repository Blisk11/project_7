import streamlit as st
from streamlit_metrics import metric, metric_row
import pandas as pd
from toolbox import predict
from PIL import Image
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
import numpy as np


# streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Credit Default App', layout = 'wide', initial_sidebar_state = 'auto')
#data_directory = r'C:\Users\juvaugha\Documents\PYTHON\OPCR PROJECTS\Project_7\simplified_app\data/'
# import df and model
model = predict.import_model()
test_df = predict.import_test_df()
train_df = predict.import_train_df()
model_features_with_target = model.feature_name_
model_features_with_target.append('TARGET')
suggested_initial_cut_off_test = predict.initial_cut_off_suggestion(test_df[model_features_with_target] , model)
suggested_initial_cut_off_train = predict.initial_cut_off_suggestion(test_df[model_features_with_target] , model)

# Home page aka project brief

image = Image.open(requests.get("https://www.batangasjobs.net/jobscloud?format=nothtml&cId=MjEzMDU", stream=True).raw)

st.sidebar.image(image, width= 200, )

#Master side bar
st.sidebar.title("Selection Panel")
page_names = ['Project Brief',  'Model inputs for cut off explained', 'Customer Dashboard']

page = st.sidebar.radio('Navigation pages', page_names)

if page == 'Project Brief':
    
    st.title('Project Brief')
    st.header('Credit default machine learning app')
    st.markdown(" #### For project number 7 of my [Data Scientist Degree](https://openclassrooms.com/en/paths/164-data-scientist#path-tabs), \
         we were tasked with deploying a machine learning model online. The dataset is from this \
            [Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk) ")
    st.write("")
    st.markdown(" #### ** The main goals of the project are: ** \n"
    "* Deploy a machine learning model online that will predict the probability of a customer defaulting on their loan \n"
    "* Create a dashboard for the banks relationship managers, in order to interpret the model \n" 
    "* [A detailed report of the project](https://www.dropbox.com/s/vv9x4t8dn2czp0w/Project%207%20Report.pdf?dl=0) " )


    #st.subheader("Table of content")
    st.markdown(""" #### **Table of content**  
    * **Model Dashboard:** An overview of the model and my custom threshold function to maximise profits  
    * **Customer Dashboard:** A dashboard destined for the bank relationship managers
    """)
    st.write("")
    st.markdown( " #### ** Some information on the dataset before you go! ** \n")

    
    
    st.markdown(" The data was provided by [Home Credit Group](https://www.homecredit.net/about-us.aspx) they operate mostly in Asia. \
        Our training dataset is mostly [Cash loans](https://www.homecredit.net/about-us/our-products.aspx). ")
    st.write("")

    st.markdown(  
    """ We do not know if the data is from one country or multiple, so we don't know if the currency value is homogenous throughout the dataset. A reverse google image
        search of the picture in the KAGGLE competition, seems to indicate it's from their Vietnamese branch. If the entire dataset is in * Vietnamese dong *, 
        then the maximum loan of our dataset is about 23 USD$. Most of the information on the loan purpose were missing, but when they were indicated they were mostly for
        consumer goods. Most common: Mobile phones, electronics, computers, furniture. ** The data with previous loans information seems to indicate the average return per credit is ~45%. **
                   
       """) 



# Customer Dashboard page
if page == 'Customer Dashboard':
    st.sidebar.write('***')
    st.sidebar.title('Required inputs')
    #side bar inputs
    perc_reimbursed = st.sidebar.slider(label="Average return before default (what percentage of the initial credit was repaid)", 
    min_value = 0.05, max_value = .90, value = .65, step=.05,format="%.2f")
    perc_profit = st.sidebar.slider(label='Expected Return, as a percentage of the loan (interest rate)' , 
    min_value = 0.01, max_value = 0.65, value = .15, step=.01,format="%.2f")

    customer_id = st.sidebar.selectbox("Please select a Customer ID", test_df.index.sort_values().to_list(), index = 4966)

    information_ok_customer = st.empty()
    information_ok_customer = st.sidebar.checkbox("Confirm selection?")

    # Main page
    st.title('Customer Dashboard')

    if information_ok_customer != True:
        st.subheader('Customer group information')
        metric_row(
    {
        'Average credit: ': "{:0,.0f}".format(int(test_df.AMT_CREDIT.mean())) , 
       'Minimum credit: ': "{:0,.0f}".format(int(test_df.AMT_CREDIT.min())) ,
        'Maximum credit: ': "{:0,.0f}".format(int(test_df.AMT_CREDIT.max())) ,
    }
)
        st.write('Average credit: ', "{:0,.0f}".format(int(test_df.AMT_CREDIT.mean())))
        st.write('Minimum credit: ', "{:0,.0f}".format(int(test_df.AMT_CREDIT.min())))
        st.write('Maximum credit: ', "{:0,.0f}".format(int(test_df.AMT_CREDIT.max())))
        st.write("*** \n")

        st.header('You can change the information in the side bar')
        st.markdown(""" 
        * **Average return before default: ** For the group information above, what is the average percentage of the credit that is reimbursed before default? 
        * ** Please select a Customer ID: ** The customer ID you wish to evaluate.
        * **Expected Return, as a percentage of the loan: ** The total payments (interest, insurance, fees) divided by the loan.
        """)
        st.subheader('Please confirm your selection in the side bar')
    else:
        st.subheader('Selection Recap')
        metric_ID = st.empty()
        with metric_ID:
            metric("Customer ID", customer_id)
       
        metric_row(
    {
        "Credit requested": int(test_df.loc[customer_id]['AMT_CREDIT']),
        "Average Percentage of Credit reimbursed": str(int(perc_reimbursed*100)) + '%',
        "Expected interest on loan": str(int(perc_profit* test_df.loc[customer_id]['AMT_CREDIT'])),
    }
)
        st.write('***')
        # Customer prediction
        prediction = model.predict_proba(test_df[model.feature_name_].loc[[customer_id]])[0][1]
        # Get the updated optimal cutoff on the validation dataset
        test_opt_perc, test_plot_dic = predict.find_optimal_proba (test_df, model, suggested_initial_cut_off_test, perc_reimbursed, perc_profit)

        # print the result
        st.subheader('Model prediction')
        
        st.write('Customer Score: ', int((prediction)*100), ' %') 
        st.write('Cut of score: ', int(test_opt_perc*100), ' %' )
        if test_opt_perc >= (prediction):
            st.markdown(" #### ** Customer Credit application is approved! :trophy: :moneybag: ** ")
            st.markdown('##### If you want an example of a default, ** customer ID: 454442 **, has the highest probability of default')
        else: 
            st.markdown(" #### ** Unfortunately your customer application has been denied. ** ")



        # !! Shap explainer and shap_values
        explainer = shap.TreeExplainer(model, num_iteration=model.best_iteration_)
        shap_values = explainer.shap_values(test_df[model.feature_name_].loc[[customer_id]])
        #shap_values = explainer.shap_values(train_df[model.feature_name_].sample(int(.25* len(train_df))).append(test_df[model.feature_name_].loc[customer_id]))

        # show customer data as a dataframe, top shap features
        st.write('***')
        st.subheader('Additional customer information')

        st.write("Let's have a look at how the model scored the most impactful features")
        plt.title('Model decision process')

        # make the shap plot smaller
        col1, col2 = st.columns(2)
        shap.decision_plot(explainer.expected_value[1], shap_values[1][-1], 
        test_df[model.feature_name_].loc[customer_id])
        col1.pyplot()

        predict.st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][-1], 
        test_df[model.feature_name_].loc[customer_id], feature_names= model.feature_name_, link='logit'))

        st.write("Absolute percentile compared to training data (in order of model importance for the selected customer), use scroll bar to view the whole table.")
        top_total_features_list = predict.get_feature_importance(model, shap_values, len(model.feature_name_))
        # show customer data as percentile of training data
        percentile_dict = {}
        for col in model.feature_name_:
            percentile = stats.percentileofscore((train_df[col]), (test_df.loc[customer_id][col]), 'mean')
            if percentile is np.nan:
                percentile_dict[col] = percentile, test_df.loc[customer_id][col], train_df[col].min(), train_df[col].max()
            else:
                percentile_dict[col] = str(int(percentile))+'%' , test_df.loc[customer_id][col], train_df[col].min(), train_df[col].max()

        percentile_df = pd.DataFrame.from_dict(percentile_dict, orient='index')
        percentile_df.reset_index(inplace=True)
        percentile_df.columns = ['Feature Name', 'Percentile', 'Customer Value', 'Training Data Minimum', 'Training Data Maximum']
        percentile_df = percentile_df.set_index('Feature Name').reindex(index= top_total_features_list)

        feature_information_df = predict.import_feature_information_df(data_directory)
        #percentile_df = pd.merge(percentile_df, feature_information_df, left_on = 'Feature Name', right_on = 'col_name')
        col1, col2 = st.columns(2)
        st.dataframe(percentile_df)
        st.write('Short description of the features below')
        feature_deep_dive_list = st.multiselect('Select features to see their descriptions', percentile_df.index)
        for feature in feature_deep_dive_list:
            st.write(feature)
            st.write(feature_information_df[feature_information_df['col_name']== feature]['Description'])
            st.write()


if page == 'Model inputs for cut off explained':
    
    st.sidebar.write('***')
    st.sidebar.title('Required inputs')
    #side bar inputs
    perc_reimbursed = st.sidebar.slider(label="Average return before default (what percentage of the initial credit was repaid)", min_value = 0.05, max_value = .90, value = .65, step=.05,format="%.2f")
    perc_profit = st.sidebar.slider(label='Expected Return, as a percentage of the loan (interest rate)' , 
    min_value = 0.01, max_value = 0.65, value = .15, step=.01,format="%.2f")
    information_ok_dashboard = st.empty()
    information_ok_dashboard = st.sidebar.checkbox("Confirm selection?")

    #required variables
    
    test_opt_perc, test_plot_dic = predict.find_optimal_proba (test_df[model_features_with_target], model, suggested_initial_cut_off_test, perc_reimbursed, perc_profit)
    train_opt_perc, train_plot_dic = predict.find_optimal_proba (train_df[model_features_with_target], model, suggested_initial_cut_off_train, perc_reimbursed, perc_profit)

    
    st.title('Model Dashboard')
    
    st.subheader("Inputs for Threshold explained")
    st.markdown(''' This problem is a imbalance binary classification problem. So we train the model to maximise the area under the [ROC curve]
    (https://en.wikipedia.org/wiki/Receiver_operating_characteristic). The curve is useful to understand the trade-off in the true-positive rate and
     false-positive rate for different thresholds. * 0= good customer, 1 = Default, *Our aim is to "catch" every defaulting customer, 
     even if that means we deny loans to customer who would have been able to repay their loan because the cost of one customer is equal to the profits 
     generated by many "good customers". 
     ''') 
    r1_three_col1, r1_three_col2 = st.columns(2)
    predict.print_roc_curve(train_df, model, 'Training Data ROC curve', r1_three_col1)
    predict.print_roc_curve(test_df, model, 'Validation Data ROC curve', r1_three_col2)

    
    st.markdown('''
    Our model outputs a probability between 0 and 1. Closer it is to 0, the more our model is confident a customer will repay their loan.
    Now we could simply define the threshold of acceptance at 0.50, but there are more sophisticated techniques. For example,
    The * Validation threshold suggestion * above was calculated using the [Geometric mean]
    (https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/#:~:text=There%20are%20many%20ways,for%20each%20threshold%20directly.).   
      
    But for this perticular problem, we can optimise the cutoff threshold to ** maximise profits **. Since the profits/ cost of our 0 and 1 should be available.
    Unfortunately, we did not have access to these variables in the labeled dataset of the competition, hence why I added the inputs which are then passed on to 
    each previous loan. It is a bit unfair since theses variables would have a significant effect on the target variable. Nevertheless, since we seperated the dataset
    into categories based on the amount of credit requested, and that these credits are for consumer products, the interest range should be fairly small. 
    Choose some variables in the side bar and we will visualize how the effect they have on our optimal threshold.
    ''') 

    if information_ok_dashboard != True:
        st.subheader("Confirm your selection in the side bar to continue!")
        # required variables
        # Get the updated optimal cutoff on the validation dataset
    else:
        st.markdown(' #### ** Choose different variable and see how it affects the confusion matrix on the left. Compare with the right matrix with the Optimal [Geometric Mean]\
        (https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/#:~:text=There%20are%20many%20ways,for%20each%20threshold%20directly.) **')
        
        test_opt_perc, test_plot_dic = predict.find_optimal_proba (test_df[model_features_with_target], model, suggested_initial_cut_off_test, perc_reimbursed, perc_profit)
        train_opt_perc, train_plot_dic = predict.find_optimal_proba (train_df[model_features_with_target], model, suggested_initial_cut_off_train, perc_reimbursed, perc_profit)
        r2_three_col1, r2_three_col2 = st.columns(2)
        predict.print_confusion_matrix(test_df[model_features_with_target], model, test_opt_perc, 'Custom cut off based on your inputs', r2_three_col1 )
        predict.print_confusion_matrix(test_df[model_features_with_target], model, suggested_initial_cut_off_test, 'Initial model cut off confusion matrix', r2_three_col2 )
        
        st.markdown(" #### ** Our goal is to maximise profits! :dollar: :moneybag: :dollar: **\
         The optimal binary classification matrix is the one that will generate the most money on our validation data! ")

        st.markdown(" ##### Let's instead look at ** average revenue per customer ** based on your selection compared to the suggested cut-off.")
        st.markdown(' ** Change the settings in the side bar to see how the graph changes based on the inputs! **')
        st.write('Average credit: ', int(test_df.AMT_CREDIT.mean()))
        st.write('Suggested cut off: ', suggested_initial_cut_off_test.round(4))
        st.write('Based on inputs optimal cut off: ', test_opt_perc.round(4))
        col1, col2 = st.columns(2)
        predict.print_return_graph('Training Data', train_opt_perc, train_plot_dic, suggested_initial_cut_off_train, col1)
        predict.print_return_graph('Validation Data', test_opt_perc, test_plot_dic, suggested_initial_cut_off_test, col2)

        st.write('')
        

        st.markdown("* ** The X axis is the custom cut off point, a customer needs to be below the cut-off to receive a loan ** \n"
        "* ** The Y axis is the total returns divide by the number of customer, to calculate this we look at the * outcome * of the cut off point: ** \n"
        "\n" 
        "* ** Bad loans that receive a credit = loss calculated ** \n" 
        "* ** Bad loans that are denied a credit = 0 **\n" 
        "* ** Good loans that receive a credit = profits generated by loan ** \n" 
        "* ** Good loans that are denied = 0 ** \n"     
        "* ** Then we take the sum of all outcomes and divide it by the number of applicants to get our average revenue per customer** \n"      
        )

        st.markdown("* #### ** So to recap ** \n"
        "* ** We train our model with the standard *auc * metric ** \n"
        "* ** Then on our validation dataset, we run a simulation on different cut off points  ** \n" 
        "* ** We compute the results based on the formula stated above or even better we use actual returns and cost \n ** " 
        "* ** We then find the optimal cut off point which will use when running the model on new applicants! ** \n"
              )
        st.markdown(' ##### You can read a more thorough explanation with this [link](https://www.dropbox.com/s/vv9x4t8dn2czp0w/Project%207%20Report.pdf?dl=0)!  :notebook: ')

