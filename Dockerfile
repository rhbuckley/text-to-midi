FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# set the HF_HOME to be a volume
ENV HF_HOME=/huggingface

# Install OS-level dependencies
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fluidsynth \
    libasound-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# set working directory to be root
WORKDIR /

# copy all files (w.r.t. .dockerignore)
COPY src/ requirements.txt ./

# install uv and create a venv
RUN pip install uv && uv venv

# update the path to include the venv
ENV PATH="/.venv/bin:$PATH"

# install the dependencies
RUN uv pip install -r requirements.txt

# download the mistral model
# RUN python -c "from unsloth import FastLanguageModel; \
#                FastLanguageModel.from_pretrained('mistralai/Mistral-7B-v0.1', load_in_4bit=False)"

# run the handler
CMD ["python", "-m", "src.deploy.handler"]

