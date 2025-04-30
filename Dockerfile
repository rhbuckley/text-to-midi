FROM python:3.10-slim

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r ./requirements-runpodgpt2.txt

CMD ["python3", "-m", "rp_deploy.handler_gpt2"]

