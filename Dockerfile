FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone SuperGlue manually (no setup.py)
RUN git clone --depth 1 \
    https://github.com/magicleap/SuperGluePretrainedNetwork.git \
    /app/SuperGluePretrainedNetwork

# Add to Python path
ENV PYTHONPATH="/app/SuperGluePretrainedNetwork:${PYTHONPATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
