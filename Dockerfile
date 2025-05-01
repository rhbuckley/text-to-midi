FROM continuumio/miniconda3

# set working directory to be root
WORKDIR /

# copy all files (w.r.t. .dockerignore)
COPY . .

# create conda environment
RUN conda env create -f environment-deploy.yml

# run the handler
# TODO: move this to src with multiple handlers
# CMD ["python3", "-m", "rp_deploy.handler_gpt2"]
CMD ["conda", "run", "-n", "text2midi", "python", "-m", "src.deploy.handler"]

