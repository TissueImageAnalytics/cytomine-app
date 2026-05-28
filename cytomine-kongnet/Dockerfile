FROM python:3.10.14-slim-bullseye

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

# Install tiatoolbox, pytorch, and dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

COPY ./models/ /models
COPY ./scripts/run.py /app/run.py

ENTRYPOINT ["python", "run.py"]
