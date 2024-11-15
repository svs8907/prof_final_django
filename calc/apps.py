from django.apps import AppConfig
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load the disease data from CSV
disease_data = pd.read_csv('disease.csv')

@app.route('/get_disease_data', methods=['GET'])
def get_disease_data():
    disease = request.args.get('disease')
    # Filter data for the specified disease
    filtered_data = disease_data[disease_data['disease_name'] == disease]

    if filtered_data.empty:
        return jsonify({'error': 'Disease not found'}), 404

    # Format data as JSON
    result = {
        'mortality_rate': filtered_data[['day', 'mortality_rate']].values.tolist(),
        'immunization_rate': filtered_data[['year', 'immunization_rate']].values.tolist(),
        'age_distribution': filtered_data[['age_group', 'percentage']].values.tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


class CalcConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'calc'
