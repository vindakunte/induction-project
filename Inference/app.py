from flask import Flask, request, jsonify
from marshmallow import Schema, fields, ValidationError, EXCLUDE
from datetime import datetime
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from input_preprocessing import LeverageCalculator
from input_preprocessing import AgeCalculator
from input_preprocessing import CategoryEncoder
from input_preprocessing import TextPreprocessor
from input_preprocessing import TfidfConcatenator
 
with open('loan_model_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
 
app = Flask(__name__)
 
df = pd.read_csv('training_data.csv', index_col=False)
feature_names = df.columns
training_data = df[feature_names].values
 
explainer = LimeTabularExplainer(
    training_data= training_data,
    feature_names=feature_names,
    training_labels=['0', '1'],
    mode='classification'
)
 
class ModelRequestSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    loan_amnt = fields.Float(required=True, data_key='loanAmnt')
    emp_length = fields.Integer(required=True, data_key='empLength')
    annual_inc = fields.Float(required=True, data_key='annualInc')
    delinq_2yrs = fields.Integer(required=True, data_key='delinq2yrs')
    inq_last_6mths = fields.Integer(required=True, data_key='inqLast6mths')
    mths_since_last_delinq = fields.Integer(required=True, data_key='mthsSinceLastDelinq')
    mths_since_last_record = fields.Integer(required=True, data_key='mthsSinceLastRecord')
    open_acc = fields.Integer(required=True, data_key='openAcc')
    pub_rec = fields.Integer(required=True, data_key='pubRec')
    revol_bal = fields.Integer(required=True, data_key='revolBal')
    revol_util = fields.Float(required=True, data_key='revolUtil')
    total_acc = fields.Integer(required=True, data_key='totalAcc')
    earliest_cr_line = fields.DateTime(format='%Y-%m-%d %H:%M:%S', allow_none=True, data_key='earliestCrLine')
    purpose = fields.String(required=True, data_key='purpose')
    desc = fields.String(required=False, allow_none=True, data_key='desc')
 
@app.route('/application', methods=['POST'])
def handle_application():
    json_data = request.get_json()
    print("Received JSON data:", json_data)

    # Preprocess the 'earliestCrLine' field
    if json_data['earliestCrLine'] is not None:
        date_parts = json_data['earliestCrLine']
        try:
            json_data['earliestCrLine'] = datetime(*date_parts).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            return jsonify({"error": "Invalid date format in 'earliestCrLine'", "details": str(e)}), 400

    print("Processed JSON data:", json_data)
    schema = ModelRequestSchema()

    try:
        data = schema.load(json_data)
    except ValidationError as err:
        print("Validation error:", err.messages)
        return jsonify({"error": "Validation failed", "details": err.messages}), 400

    print("Data loaded:", data)
    response = process_data(data)

    return jsonify(response)
 
def process_data(data):
    print(training_data.shape[1])
    print("Obtained data:", data)
    input_data = pd.DataFrame([data])

    print("Input data shape:", input_data.shape[1])
 
    transformed_data = pipeline.named_steps['leverage_calculator'].transform(input_data)
    print("Transformed data shape:", transformed_data.shape[1])

    transformed_data = pipeline.named_steps['age_calculator'].transform(transformed_data)
    print("Transformed data shape:", transformed_data.shape[1])

    transformed_data = pipeline.named_steps['category_encoder'].transform(transformed_data)
    print("Transformed data shape:", transformed_data.shape[1])

    transformed_data = pipeline.named_steps['text_preprocessor'].transform(transformed_data)
    print("Transformed data shape:", transformed_data.shape[1])

    transformed_data = pipeline.named_steps['tfidf_concat'].transform(transformed_data)
    print("Transformed data shape:", transformed_data.shape[1])

    transformed_data = pipeline.named_steps['scaler'].transform(transformed_data)
    print("Transformed data shape:", transformed_data.shape[1])

 
    print( feature_names)
    print(feature_names.size)
    print("Transformed data shape:", transformed_data.shape[1])
    #print(transformed_data.shape[1])

    prediction = pipeline.named_steps['model'].predict(transformed_data)[0]
    prediction_proba = pipeline.named_steps['model'].predict_proba(transformed_data)[0]
    print(prediction_proba)
    print(prediction)
    status = ""
    if(prediction_proba[1]*1000 > 340.2045739512086):
        status = "Approved"
   
    else:
        status = "Declined"
 
 
    exp = explainer.explain_instance(
        transformed_data[0],
        pipeline.named_steps['model'].predict_proba,
        num_features=45
    )
 
    explanation = exp.as_list() 
    unscaled_explanation = []
    desired_feature_names = ['loan_amnt', 'leverage_ratio', 'emp_length', 'annual_inc', 'delinq_2yrs',
                            'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
                            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'earliest_cr_line', 'purpose']


    for feature, weight in explanation:
        if '>' in feature or '<=' in feature:
            feature_name = feature.split(' ')[0]
            threshold = float(feature.split(' > ')[-1].split(' <= ')[-1])
            isGreater = '>' in feature

            if feature_name in desired_feature_names:
                feature_idx = feature_names.get_loc(feature_name)
                std = pipeline.named_steps['scaler'].scale_[feature_idx]

                original_threshold = (threshold * std)
                unscaled_explanation.append((feature_name, original_threshold, weight, isGreater))

    unscaled_explanation = unscaled_explanation[:5]
    
    if status == "Approved":
        explanation_str = ''
    else:
        explanation_str = "; ".join([f"{feature} {'>' if isGreater else '<='} {threshold}: {weight:.4f}" 
                                for feature, threshold, weight, isGreater in unscaled_explanation])    
    
    result = {
        "status": status,
        "score": prediction_proba[1]*1000,
        "reason": explanation_str
    }
    return result
 
if __name__ == '__main__':
    app.run(host='localhost', port=7778)
 