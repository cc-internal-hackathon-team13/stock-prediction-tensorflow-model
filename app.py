from model import model
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    result = model.predict()
    return jsonify({"next_day_prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
