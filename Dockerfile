from nvidia/cuda:12.1.0-runtime-ubuntu22.04

# install python
RUN apt-get update && apt-get install -y python3 python3-pip git curl

# working directory
WORKDIR /workspace

# copy project files
COPY requirements.txt .
COPY pyproject.toml .

# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# create data dir
RUN mkdir -p data checkpoints outputs

# default command
CMD ["python3", "src/train.py"]