FROM python:3.12-slim 
#Base image

COPY . /app
RUN ls -al /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "shapedetector.py"]