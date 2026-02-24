from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict
import os

app = Flask(__name__)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'image field is required'}), 400
    results = predict(data['image'])
    return jsonify({'predictions': results})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)