# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("📊 Claims Daily Analysis Dashboard")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload a .csv file", type=["csv"])

def extract_file_date(filename: str) -> str:
    digits = ''.join(filter(str.isdigit, filename))
    if len(digits) >= 4:
        mmdd = digits[-4:]
        try:
            parsed_date = datetime.strptime(mmdd, "%m%d").replace(year=datetime.now().year)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None

def process_dataframe(df, source_file):
    # Convert types
    if 'claim_age' in df.columns:
        df['claim_age'] = pd.to_numeric(df['claim_age'], errors='coerce')
    if 'clm_totalamt' in df.columns:
        df['clm_totalamt'] = pd.to_numeric(df['clm_totalamt'], errors='coerce')
    if 'claim_date' in df.columns:
        df['Claim_date'] = pd.to_datetime(df['Claim_date'], errors='coerce')

    # Add file_date from filename
    file_date = extract_file_date(source_file)
    df['file_date'] = file_date
    df['source_file'] = source_file

    # Bucket claim_age
    if 'claim_age' in df.columns:
        bins = [-1, 4, 9, 14, 19, 24, 29, float('inf')]
        labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30+']
        df['claim_age_bucket'] = pd.cut(df['claim_age'], bins=bins, labels=labels)

    return df

if uploaded_file:
    with st.spinner("Reading and processing file..."):
        try:
            df = pd.read_csv(uploaded_file)
            df = process_dataframe(df, uploaded_file.name)

            st.success("File loaded and processed ✅")

            # --- Filters ---
            st.sidebar.header("Filters")
            state_options = df['State_Check'].dropna().unique().tolist() if 'State_Check' in df.columns else []
            lob_options = df['Corrected_LOB'].dropna().unique().tolist() if 'Corrected_LOB' in df.columns else []
            bucket_options = df['claim_age_bucket'].dropna().unique().tolist() if 'claim_age_bucket' in df.columns else []
            workability_options = df['Workability'].dropna().unique().tolist() if 'Workability' in df.columns else []

            selected_workability = st.sidebar.multiselect("Workability", workability_options, default=workability_options)
            selected_states = st.sidebar.multiselect("State", state_options, default=state_options)
            selected_lobs = st.sidebar.multiselect("LOB", lob_options, default=lob_options)
            selected_buckets = st.sidebar.multiselect("Claim Age Bucket", bucket_options, default=bucket_options)

            if state_options and lob_options and bucket_options:
                filtered_df = df[
                    df['State_Check'].isin(selected_states) &
                    df['Corrected_LOB'].isin(selected_lobs) &
                    df['claim_age_bucket'].isin(selected_buckets)&
                    df['Workability'].isin(selected_workability)
                ]
            ## --- Descriptive Summary Cards ---
                st.subheader("📌 At-a-Glance Summary")
                total_claims = filtered_df.shape[0]
                total_amount = filtered_df['clm_totalamt'].sum()
                oldest_date = filtered_df['Claim_date'].min()

                if pd.notnull(oldest_date):
                    oldest_row = filtered_df[filtered_df['Claim_date'] == oldest_date].iloc[0]
                    oldest_state = oldest_row['State_Check']
                    oldest_lob = oldest_row['Corrected_LOB']
                else:
                    oldest_state = oldest_lob = "N/A"

                try:
                    formatted_date = pd.to_datetime(oldest_date).strftime("%Y-%m-%d")
                except Exception:
                    formatted_date = "N/A"

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Claims", f"{total_claims:,}")
                col2.metric("Total Amount ($)", f"{total_amount:,.2f}")
                col3.metric("Oldest Claim Date", formatted_date)
                col4.metric("Oldest in", f"{oldest_state} / {oldest_lob}")

                st.subheader("🔹 LOB Summary Table")
                summary = (
                    filtered_df.groupby([ 'Corrected_LOB', 'claim_age_bucket'])
                    .agg(
                        claim_count=('claim_age', 'count'),
                        total_amount=('clm_totalamt', 'sum'),
                        oldest_claim_date=('Claim_date', 'min')
                    )
                    .reset_index()
                    .sort_values(by=['Corrected_LOB', 'claim_age_bucket'])
                )
                st.dataframe(summary, use_container_width=True)
                
                st.subheader("🔹 Summary Table")
                summary = (
                    filtered_df.groupby(['State_Check', 'Corrected_LOB', 'claim_age_bucket'])
                    .agg(
                        claim_count=('claim_age', 'count'),
                        total_amount=('clm_totalamt', 'sum'),
                        oldest_claim_date=('Claim_date', 'min')
                    )
                    .reset_index()
                    .sort_values(by=['State_Check', 'Corrected_LOB', 'claim_age_bucket'])
                )
                st.dataframe(summary, use_container_width=True)

                # --- Charts ---
                st.subheader("📈 Interactive Charts")

                col1, col2 = st.columns(2)

                with col1:
                    fig1 = px.bar(
                        summary,
                        x='claim_age_bucket',
                        y='claim_count',
                        color='Corrected_LOB',
                        barmode='group',
                        title="Claim Count by Age Bucket and LOB"
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    fig2 = px.bar(
                        summary,
                        x='claim_age_bucket',
                        y='total_amount',
                        color='Corrected_LOB',
                        barmode='group',
                        title="Total Claim Amount by Age Bucket and LOB"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("🕓 Oldest Claims per Group")
                fig3 = px.scatter(
                    summary,
                    x='oldest_claim_date',
                    y='claim_age_bucket',
                    color='Corrected_LOB',
                    symbol='State_Check',
                    title="Oldest Claim Dates by Bucket and LOB"
                )
                st.plotly_chart(fig3, use_container_width=True)

                st.subheader("🔍 Raw Filtered Data")
                st.dataframe(filtered_df, use_container_width=True)

                # --- Download ---
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Filtered Data as CSV", csv, "filtered_claims.csv", "text/csv")
            else:
                st.warning("Required columns are missing in the uploaded file. Please check the file format.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")