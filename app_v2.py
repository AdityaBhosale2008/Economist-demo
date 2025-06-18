import streamlit as st 
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns



# -----------------------------------------------------------------------------------------------------------------------------------
#  Streamlit 
# -----------------------------------------------------------------------------------------------------------------------------------


# Sample dataset (or load your real one)
@st.cache_data
def load_data():
    return pd.read_csv("data/baseline_model.csv")  # Replace with your dataset

# Set layout
st.set_page_config(page_title="Churn App", layout="wide")
st.title("📕 Economist Churn App")
st.markdown("This app predicts whether a customer will churn based on their engagement, behavior, and subscription data.")


# Tabs
tab1, tab2 = st.tabs(["🔍 Predict Churn", "📊 Feature Summary Dashboard"])

# ========== TAB 1: Churn Prediction ==========
with tab1:
    # === All previous prediction UI & logic goes here ===
    st.subheader("Churn Prediction Form")
    # Sidebar header
    st.sidebar.header("📋 User Input Options")

    # Input Sections
    with st.sidebar.expander("📦 Subscription Details", expanded=True):
        subscription_type = st.selectbox('Subscription Plan Type', ['Espresso', 'Digital', 'Digital+Print'])
        plan_type = st.selectbox('Plan Type', ['Monthly', 'Yearly'])
        signup_source = st.selectbox('Signup Source', ['Web', 'Mobile App', 'Referral'])
        st.markdown("**👇 Check the box if answer is Yes ✅**")
        auto_renew = st.checkbox("Auto-renew Enabled?")
        discount_used_last_renewal = st.checkbox("Discount Used at Last Renewal?")
        previous_renewal_status = st.checkbox("Was Previously Renewed?")
        downgrade_history = st.checkbox("Was Downgraded Before?")

    with st.sidebar.expander("🌍 User Profile"):
        region = st.selectbox('Region', ['North America', 'Europe', 'Asia', 'Other'])
        primary_device = st.selectbox('Primary Device', ['Tablet', 'Mobile', 'Desktop'])
        payment_method = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'PayPal'])
        most_read_category = st.selectbox('Most Read Category', ['Technology', 'Business', 'Science', 'Health', 'Politics', 'Entertainment', 'Culture'])
        last_campaign_engaged = st.selectbox('Last Campaign Engaged', ['Newsletter Promo', 'Retention Offer', 'Survey'])

    with st.sidebar.expander("📊 Engagement Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            customer_age = st.slider('Customer Age', 18, 100, 25)
            avg_articles_per_week = st.slider('Avg Articles/Week', 0.0, 9.0, 0.0, 0.1)
            article_skips_per_week = st.slider('Article Skips/Week', 0, 10, 0)
            days_since_last_login = st.slider('Days Since Last Login', 0, 100, 0)
        with col2:
            support_tickets_last_90d = st.slider('Support Tickets (Last 90 Days)', 0, 10, 0)
            email_open_rate = st.slider('Email Open Rate', 0.00, 1.00, 0.00, 0.01)
            time_spent_per_session_mins = st.slider('Time/Session (mins)', 0.0, 30.0, 0.0, 0.1)
            tenure_days = st.slider('Tenure (Days)', 0, 1825, 25)

    with st.sidebar.expander("📈 Engagement Scores"):
        col1, col2 = st.columns(2)
        with col1:
            completion_rate = st.slider('Completion Rate', 0.00, 1.00, 0.00, 0.01)
            campaign_ctr = st.slider('Campaign CTR', 0.00, 1.00, 0.00, 0.01)
        with col2:
            nps_score = st.slider('NPS Score', -100, 100, 0)
            sentiment_score = st.slider('Sentiment Score', -1.5, 1.5, 0.0, 0.1)
            csat_score = st.slider('CSAT Score (1-5)', 1, 5, 3)

    # Prediction button
    st.markdown("---")
    # prediction_button = st.button('🔍 Predict Churn', type="primary")
    col1, col2 = st.columns([1, 1])
    predict_button = col1.button("🔍 Predict", use_container_width=True)
    clear_button = col2.button("🗑️ Clear Inputs", use_container_width=True)



with tab2:
    st.markdown("## 📊 Feature Summary Dashboard")
    st.write("Explore feature distributions, statistics, and insights")

    data = load_data()  
    feature = st.selectbox("🔎 Select a feature to analyze", data.columns)

    if pd.api.types.is_numeric_dtype(data[feature]):
        st.markdown("### 📌 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{data[feature].mean():.2f}")
        col2.metric("Median", f"{data[feature].median():.2f}")
        col3.metric("Std. Dev", f"{data[feature].std():.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Min", f"{data[feature].min():.2f}")
        col5.metric("Max", f"{data[feature].max():.2f}")
        col6.metric("Missing", data[feature].isna().sum())

        col7, col8 = st.columns(2)
        with col7:
            st.markdown("### 📈 Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data[feature], kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)

        with col8:
            st.markdown("### 📦 Boxplot")
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            sns.boxplot(x=data[feature], color='lightgreen', ax=ax2)
            st.pyplot(fig2)

        if 'churn' in data.columns:
            with st.expander("🔍 Compare by Churn"):
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                sns.kdeplot(data=data, x=feature, hue='churn', fill=True, palette='Set2', ax=ax3)
                st.pyplot(fig3)

    else:
        st.markdown("### 🧮 Categorical Summary")
        st.write(data[feature].value_counts())

        fig_cat, ax_cat = plt.subplots(figsize=(8, 4))
        sns.countplot(data=data, x=feature, palette='pastel', ax=ax_cat)
        ax_cat.set_xticklabels(ax_cat.get_xticklabels(), rotation=45)
        st.pyplot(fig_cat)

        if 'churn' in data.columns:
            with st.expander("🔍 Compare by Churn"):
                fig_cat_churn, ax_cat_churn = plt.subplots(figsize=(8, 4))
                sns.countplot(data=data, x=feature, hue='churn', palette='Set2', ax=ax_cat_churn)
                ax_cat_churn.set_xticklabels(ax_cat_churn.get_xticklabels(), rotation=45)
                st.pyplot(fig_cat_churn)

    with st.expander("🧮 Correlation Matrix"):
        corr = data.select_dtypes(include='number').corr()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

    with st.expander("📷 Pairplot (Sample)"):
        sample = data.select_dtypes(include='number').sample(n=min(200, len(data)), random_state=42)
        fig6 = sns.pairplot(sample)
        st.pyplot(fig6)
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

  
def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model      


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

if clear_button:
    st.experimental_rerun()

if predict_button:
    df = preprocess(user_input)
    model = load_xgb_model()
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"
    st.success(f"📊 **Prediction:** {result}")
    st.info(f"🧠 Model Confidence: **{proba * 100:.2f}%** for Churn")

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    st.subheader("🔎 Feature Importance")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    