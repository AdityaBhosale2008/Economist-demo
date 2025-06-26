import streamlit as st 
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier




# -----------------------------------------------------------------------------------------------------------------------------------
#  Streamlit 
# -----------------------------------------------------------------------------------------------------------------------------------

st.title('📕Economist Churn Prediction')
st.write('This app predicts whether the customer will churn based on given Features.')


# Select Box
subscription_type = st.sidebar.selectbox('Select the Subscription Plan Type', ['Espresso', 'Digital', 'Digital+Print'])
plan_type = st.sidebar.selectbox('Select the Plan Type', ['Monthly', 'Yearly'])
primary_device = st.sidebar.selectbox('Select the Primary Device', ['Tablet', 'Mobile', 'Desktop'])
region = st.sidebar.selectbox('Select Region of Customer', ['North America', 'Europe', 'Asia', 'Other'])
most_read_category = st.sidebar.selectbox('Select the Most Read Category', ['Technology', 'Business', 'Science', 'Health', 'Politics', 'Entertainment', 'Culture'])
last_campaign_engaged = st.sidebar.selectbox('Select if campaign used on last renewal', ['Newsletter Promo','Retention Offer', 'Survey'])
payment_method = st.sidebar.selectbox('Select the Payment Method', ['Credit Card','Debit Card', 'PayPal'])
signup_source =  st.sidebar.selectbox('Select the signup source',['Web','Mobile App','Referral'])


# Sliders
customer_age = st.sidebar.slider('Select the Customer Age', min_value=18, max_value=100, step=1, value=18)
avg_articles_per_week = st.sidebar.slider('Select the Average Articles per Week', min_value=0.0, max_value=9.0, step=0.1, value=0.0)
article_skips_per_week = st.sidebar.slider('Select the Average Articles Skips per Week', min_value=0, max_value=10, step=1, value=0)
days_since_last_login = st.sidebar.slider('Select the Days since Last Login', min_value=0, max_value=100, step=1, value=0)
support_tickets_last_90d = st.sidebar.slider('Select the Support Tickets in Last 90 Days', min_value=0, max_value=10, step=1, value=0)
email_open_rate = st.sidebar.slider('Select the Email Open Rate', min_value=0.00, max_value=1.00, step=0.01, value=0.00)
time_spent_per_session_mins = st.sidebar.slider('Select the Time Spent per Session in Minutes', min_value=0.0, max_value=30.0, step=0.1, value=0.0)
tenure_days = st.sidebar.slider('Select the Tenure of Customer', min_value=0, max_value=1825, step=1, value=25)
completion_rate = st.sidebar.slider('Select the Completion rate', min_value=0.00, max_value=1.00, step=0.01, value=0.00)
campaign_ctr = st.sidebar.slider('Select the Campaign Score', min_value=0.00, max_value=1.00, step=0.01, value=0.00)
nps_score = st.sidebar.slider('Select the ', min_value=-100, max_value=100, step=1, value=0)
sentiment_score = st.sidebar.slider('Select the ', min_value=-1.5, max_value=1.5, step=0.1, value=0.0)
csat_score = st.sidebar.slider('Select the ', min_value=1, max_value=5, step=1, value=0)


# Checkbox
discount_used_last_renewal = st.sidebar.checkbox("Was Discount_used_last_renewal?")
auto_renew = st.sidebar.checkbox("Was Auto renewal Enabled?")
previous_renewal_status = st.sidebar.checkbox("What was Previous_renewal_status?") 
downgrade_history = st.sidebar.checkbox("Was Subscription Downgrade Before?")



# -----------------------------------------------------------------------------------------------------------------------------------
#  Prediction 
# -----------------------------------------------------------------------------------------------------------------------------------



user_input = {
    'subscription_type': subscription_type,
    'plan_type': plan_type,
    'primary_device': primary_device,
    'region': region,
    'most_read_category': most_read_category,
    'last_campaign_engaged': last_campaign_engaged,
    'payment_method': payment_method,
    'signup_source': signup_source,
    'customer_age': customer_age,
    'avg_articles_per_week': avg_articles_per_week,
    'article_skips_per_week': article_skips_per_week,
    'days_since_last_login': days_since_last_login,
    'support_tickets_last_90d': support_tickets_last_90d,
    'email_open_rate': email_open_rate,
    'time_spent_per_session_mins': time_spent_per_session_mins,
    'tenure_days': tenure_days,
    'completion_rate': completion_rate,
    'campaign_ctr': campaign_ctr,
    'nps_score': nps_score,
    'sentiment_score': sentiment_score,
    'csat_score': csat_score,
    'discount_used_last_renewal': 'Yes' if discount_used_last_renewal else 'No',
    'auto_renew': 'Auto' if auto_renew else 'Manual',
    'previous_renewal_status': 'Yes' if previous_renewal_status else 'No',
    'downgrade_history': 'Yes' if downgrade_history else 'No'
}


model_select = st.selectbox('Select Machine Learning Model',['LogisticRegression', 'XGBClassifier', 'RandomForestClassifier','DecisionTreeClassifier'])
prediction_button = st.button('Predict', type="primary")





def load_model(model_select):
    file_map = {
        'LogisticRegression': 'model/model_1.pkl',
        'XGBClassifier': 'model/model_2.pkl',
        'RandomForestClassifier': 'model/model_3.pkl',
        'DecisionTreeClassifier': 'model/model_4.pkl'
    }
    with open(file_map[model_select], 'rb') as f:
        model = pickle.load(f)
    return model


MODEL_FEATURES = ['subscription_type', 'plan_type', 'auto_renew',
       'avg_articles_per_week', 'days_since_last_login',
       'support_tickets_last_90d', 'discount_used_last_renewal',
       'email_open_rate', 'time_spent_per_session_mins',
       'completion_rate', 'article_skips_per_week',
       'previous_renewal_status', 'campaign_ctr', 'nps_score',
       'sentiment_score', 'csat_score', 'customer_age', 'signup_source',
       'downgrade_history', 'tenure_days', 'region_Asia', 'region_Europe',
       'region_North America', 'region_Others', 'most_read_Culture',
       'most_read_Environment', 'most_read_Finance', 'most_read_Politics',
       'most_read_Technology', 'primary_device_Desktop',
       'primary_device_Mobile', 'primary_device_Tablet',
       'payment_method_Credit Card', 'payment_method_Debit Card',
       'payment_method_PayPal', 'last_campaign_engaged_Newsletter Promo',
       'last_campaign_engaged_Retention Offer',
       'last_campaign_engaged_Survey']

  
      

def preprocess(user_input):
    df = pd.DataFrame([user_input])
    df['subscription_type'] = df['subscription_type'].map({'Espresso':0,'Digital':1,'Digital+Print':2})
    df['plan_type'] = df['plan_type'].map({'Monthly':0,'Annual':1})
    df['auto_renew'] = df['auto_renew'].map({'Yes':1,'No':0})
    df['discount_used_last_renewal'] = df['discount_used_last_renewal'].map({'Yes':1,'No':0})
    df['downgrade_history'] = df['downgrade_history'].map({'Yes':1,'No':0})
    df['previous_renewal_status'] = df['previous_renewal_status'].map({'Auto':1,'Manual':0})
    df['signup_source'] = df['signup_source'].map({'Web':0,'Mobile App':0,'Referral':1})

    
    df = pd.get_dummies(df, columns = ['region'], prefix = 'region')
    df = pd.get_dummies(df, columns = ['most_read_category'], prefix = 'most_read')
    df = pd.get_dummies(df, columns = ['primary_device'], prefix = 'primary_device')
    df = pd.get_dummies(df, columns = ['payment_method'], prefix = 'payment_method')
    df = pd.get_dummies(df, columns = ['last_campaign_engaged'], prefix = 'last_campaign_engaged')

    df = df.fillna(0)


    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[MODEL_FEATURES]
    return df




if prediction_button:
    st.write(user_input)
    df = preprocess(user_input)
    model = load_model(model_select)
    prediction = model.predict(df)
    proba = model.predict_proba(df)

    result = 'Churn' if prediction[0] == 1 else 'No Churn'
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {proba[0][1]*100:.2f}% for Churn")
