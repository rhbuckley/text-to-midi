FROM mambaorg/micromamba

# Install fluidsynth AND its system runtime dependencies via apt-get
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    fluidsynth \
    libsndfile1 \
    libasound2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set working directory to be root
WORKDIR /

# copy all files (w.r.t. .dockerignore)
COPY src/ src/
COPY environment-cuda.yml environment-cuda.yml

# create conda environment
RUN micromamba env create -f environment-cuda.yml

# run the handler
CMD ["micromamba", "run", "-n", "text2midi", "python", "-m", "src.deploy.handler"]

