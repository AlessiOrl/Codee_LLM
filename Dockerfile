FROM python:3.11
WORKDIR /Codee_LLM
COPY ./requirements.txt /Codee_LLM/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /Codee_LLM/requirements.txt
RUN pip install bitsandbytes>=0.42.0
RUN pip install protobuf
RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY app /Codee_LLM/app/
COPY data /Codee_LLM/data/
CMD ["fastapi", "run", "app/main.py", "--port", "80"]