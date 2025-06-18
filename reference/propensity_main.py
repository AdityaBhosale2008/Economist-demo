from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# @app.route('/predict/propensity_to_pay', methods=['POST'])
def propensity_to_pay(data):
        # data = request.get_json()
        print("Received Data:", data)
        
        balance = data.get('balance')
        household_assets = data.get('household_assets')
        mortgage_assets = data.get('mortgage_assets')
        credit_card_debt = data.get('credit_card_debt')
        income = data.get('income')
        total_credit_limit = data.get('total_credit_limit')
        ind_home_loan = data.get('ind_home_loan')
        age = data.get('age')
        gender_value = data.get('gender')
        
        zip_code = data.get('zip_code')
        wallet_size = data.get('wallet_size')
        fico_score = data.get('fico_score')
        transaction_sum = data.get('transaction_sum')
        transaction_count = data.get('transaction_count')
        contact_count = data.get('contact_count')
        online_count = data.get('Web_visits')
        debt_to_income = credit_card_debt/ (income+1)  # Avoid division by zero

        # Handle categorical variables if any (e.g., 'gender')
        # Assuming 'gender' is binary; adjust if different
        gender=1
        if gender_value == 'male':
            gender = 1
        elif gender_value == 'female':
            gender = 0
        
            # Prepare input for the model
        input_features = np.array([[balance,household_assets,mortgage_assets,credit_card_debt,income,total_credit_limit, ind_home_loan,age, gender,zip_code
                                    ,wallet_size,fico_score,transaction_sum,transaction_count,contact_count,online_count,debt_to_income]])

        print(input_features)
        # Make prediction
        ptp_model = joblib.load('propensity_to_repay_model.pkl')

        amount_due=''
        prediction = ptp_model.predict(input_features)
        if credit_card_debt>=2500:
            amount_due='High'
        else:
            amount_due='Low'
        prediction_value = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        if prediction_value == 0:
            prediction_value = 'Low'
        else:
            prediction_value = 'High'
        # Return the prediction as JSON
        print('Predicted value::',prediction_value)
        
        print('amount due::',amount_due)
        ptp_predictions={}
        ptp_predictions['propensity_to_pay']=prediction_value
        ptp_predictions['Amount_Due']=amount_due
        del ptp_model

        
        return ptp_predictions



@app.route('/predict/metrics', methods=['POST'])
def predict_metrics():

    predictions = {}
    ptp_pred=propensity_to_pay(request.get_json())

    print(ptp_pred)
    predictions['prediction_value']=ptp_pred['propensity_to_pay']
    predictions['amount_due']= ptp_pred['Amount_Due']
    

    input_data = pd.DataFrame([request.get_json()])
                                    
    input_data = input_data.drop(columns=['contact_count'])

    
    input_data['gender'] = input_data['gender'].str.lower().replace({'male': 1, 'female': 0})
    input_data['zip_code'] = pd.to_numeric(input_data['zip_code'], errors='coerce')

    # Make predictions for each model
    print("Loading Channel Model")
    clf_channel = joblib.load('clf_channel.pkl')
    predictions['predicted_channel'] = clf_channel.predict(input_data).tolist()[0]
    del clf_channel

    print("Loading RPC Model")
    reg_rpc = joblib.load('reg_rpc.pkl')
    predictions['rpc_propensity'] =  reg_rpc.predict(input_data).tolist()[0]
    del reg_rpc

    print("Loading Day of Week Model")
    clf_day = joblib.load('clf_day.pkl')
    predictions['day_of_week'] =  clf_day.predict(input_data).tolist()[0]
    del clf_day

    print("Loading Time Slot Model")
    clf_slot = joblib.load('clf_slot.pkl')
    predictions['time_slot'] =  clf_slot.predict(input_data).tolist()[0]
    del clf_slot

    
    
    return jsonify(predictions)

if __name__ == '__main__':
        app.run(debug=True)
