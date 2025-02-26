from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import math
from utils import label_encode
import json

# Initialize Flask app
app = Flask(__name__)

# Function to load model dynamically based on brand or product category
def load_model(brand, product_category):
    model_mapping = {
        'kelloggs': 'models/model1.pkl',
        'delmege': {
            'noodles': 'models/model9.pkl',
            'pasta': 'models/model10.pkl',
            'coloring': 'models/model6.pkl',
            'soya': 'models/model3.pkl',
        },
        'pakmaya': 'models/model12.pkl',
        'motha': {
            'milk shake mix': 'models/model13.pkl',
            'faluda mix': 'models/model13.pkl',
            'original jelly': 'models/model5.pkl',
            'baking powder': 'models/model7.pkl',
            'corn flour': 'models/model7.pkl', 
            'cocoa powder': 'models/model7.pkl',
            'gelatine': 'models/model7.pkl',
            'icing sugar': 'models/model7.pkl',
            'coloring': 'models/model8.pkl',
        }
    }

    if brand in model_mapping:
        if isinstance(model_mapping[brand], dict):
            if product_category in model_mapping[brand]:
                return pickle.load(open(model_mapping[brand][product_category], 'rb'))
            else:
                raise ValueError(f"No model found for product category '{product_category}' under brand '{brand}'")
        else:
            return pickle.load(open(model_mapping[brand], 'rb'))
    else:
        raise ValueError(f"No model found for brand '{brand}'")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.get_json()
    weight_unit = form_data['weight_unit']
    weight_value = form_data['weight']

    if weight_unit == 'kg':
        weight_value = float(weight_value) * 1000 
    elif weight_unit == 'ml': 
        weight_value = float(weight_value)
    elif weight_unit == 'l':
        weight_value = float(weight_value) * 1000
    elif weight_unit == 'bags':
        weight_value = float(weight_value) * 2

    input_data = pd.DataFrame({
        'Year': [form_data['Year']],
        'Month': [form_data['Month']],
        'brand': [form_data['Product Brand'].lower()],
        'sub_brand': [form_data['Sub Brand'].lower()],
        'ProductCategory': [form_data['Product Category'].lower()],
        'Channel': [form_data['Channel'].lower()],
        'flavor': [form_data['Flavor'].lower()],
        'variety': [form_data['Variety'].lower()],
        'color': [form_data['Color'].lower()],
        'Free_Issues': [form_data['Has Free Issues'].lower()],
        'Unit_Price': [form_data['unit_price']],
        'Discount': [form_data['discount']],
        'Outlet_Reach': [form_data['outlet_reach']],
        'weight': [weight_value]  
    })

    try:
        model = load_model(form_data['Product Brand'].lower(), form_data['Product Category'].lower())
        prediction = model.predict(input_data)
        predicted_qty = math.floor(prediction[0])
        total_sales_price = predicted_qty * float(form_data['unit_price'])
        return jsonify({'predicted_qty': predicted_qty, 'total_sales_price': total_sales_price})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/submit', methods=['POST'])
def submit():
    predictions = request.form.get('predictions')
    predictions = json.loads(predictions)

    total_predicted_qty = sum(p['predicted_qty'] for p in predictions)
    total_sales_price = sum(p['total_sales_price'] for p in predictions)

    return render_template('result.html', predicted_qty=total_predicted_qty, total_sales_price=total_sales_price)

if __name__ == '__main__':
    app.run(debug=True)