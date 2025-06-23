FROM python:3.8-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model
COPY inference.py .
COPY serve.py .
COPY model/ ./model/

# Expose the port the app runs on
EXPOSE 8080

# Define the entrypoint
ENTRYPOINT ["python", "serve.py"]
