FROM python:3.10

WORKDIR /

COPY . .



RUN pip install --no-cache-dir -r ./requirements-runpodgpt2.txt

ENV WANDB_API_KEY=e2058c09a28e4fb17341996da05b5f38236f031a
RUN python -m rp_deploy.get_model --project_name text2midi --model_name model_best --version v11

CMD ["python3", "-m", "rp_deploy.handler_gpt2"]

