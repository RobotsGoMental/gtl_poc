FROM huggingface/transformers-pytorch-gpu

ENV DEBIAN_FRONTEND noninteractive

# Upgrade pip
RUN pip install --upgrade pip 

# Copy requirements.txt to the /tmp directory
COPY requirements.txt /tmp/requirements.txt

#RUN python3.9 -m pip install --upgrade pip && \
    #python3.9 -m pip install --no-cache-dir -r /tmp/requirements.txt
# Install dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt