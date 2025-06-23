from flask import Flask, request, render_template
import boto3
import base64

app = Flask(__name__)

# AWS SageMaker setup
ENDPOINT_NAME = 'snake-custom-endpoint'
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get uploaded image
        file = request.files['file']
        if file:
            img_bytes = file.read()

            # Call SageMaker endpoint
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='image/jpeg',
                Body=img_bytes
            )

            result = response['Body'].read().decode('utf-8')
            prediction = result  # JSON string

    return render_template('index.html', prediction=prediction)


#Add this block to start the server
#if __name__ == '__main__':
#    app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)