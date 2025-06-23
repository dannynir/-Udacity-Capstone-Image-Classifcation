from flask import Flask, request, jsonify
from inference import model_fn, input_fn, predict_fn, output_fn

app = Flask(__name__)
model = model_fn("model")  # Relative path to model folder inside container

@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

@app.route('/invocations', methods=['POST'])
def invocations():
    content_type = request.content_type
    data = request.data
    try:
        input_data = input_fn(data, content_type)
        prediction = predict_fn(input_data, model)
        response, mime = output_fn(prediction, 'application/json')
        return response, 200, {'Content-Type': mime}
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)