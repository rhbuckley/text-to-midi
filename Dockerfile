FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# set the HF_HOME to be a volume
ENV HF_HOME=/huggingface

# Install OS-level dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    fluidsynth \
    libasound-dev \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set working directory to be root
WORKDIR /

# copy all files (w.r.t. .dockerignore)
COPY src/ src/
COPY requirements.txt requirements.txt

# install uv and create a venv
RUN pip install uv && uv venv

# update the path to include the venv
ENV PATH="/.venv/bin:$PATH"

# install the dependencies
RUN uv pip install -r requirements.txt

# download the mistral model
RUN uv pip install "huggingface_hub[hf_transfer]" "huggingface_hub[hf_xet]" && \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download(repo_id='unsloth/mistral-7b-v0.3')"

# run the handler
CMD ["python", "-m", "src.deploy.handler"]

