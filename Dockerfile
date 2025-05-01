FROM continuumio/miniconda3

# set working directory to be root
WORKDIR /

# copy all files (w.r.t. .dockerignore)
COPY . .

# create conda environment
RUN conda env create -f environment-deploy.yml

# define wandb api key (we can't do this securely bc env vars are not passed to docker)
# also, its just wandb, not the biggest deal. 
# TODO: we cannot publish this!!
ENV WANDB_API_KEY=e2058c09a28e4fb17341996da05b5f38236f031a

# download the model
# TODO: move to scritps cuz this is really a script
RUN python -m rp_deploy.get_model --project_name text2midi --model_name model_best --version v11

# run the handler
# TODO: move this to src with multiple handlers
# CMD ["python3", "-m", "rp_deploy.handler_gpt2"]
CMD ["conda", "run", "-n", "text2midi", "python", "-m", "rp_deploy.handler_gpt2"]

