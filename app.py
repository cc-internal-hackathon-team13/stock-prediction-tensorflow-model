from model import model
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def handle_prediction_data():
    result = model.predict()
    return jsonify({"next_day_prediction": result})

@app.route('/previousData', methods=['POST'])
def handle_prev_data_request():
    data = request.get_json()
    seven_day_data = model.seven_day_data(data['company'])
    return jsonify({"previousData": seven_day_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
