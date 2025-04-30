import os
import wandb
from dotenv import load_dotenv, find_dotenv


def wandb_login():
    """login to wandb -- wraps around wandb.login() with env file loading"""
    load_dotenv(find_dotenv())
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_model(
    project_name: str,
    model_name: str,
    version: str = "latest",
):
    # login to wandb
    wandb_login()

    # get the latest model from wandb's api
    api = wandb.Api()
    artifact = api.artifact(f"{project_name}/{model_name}:{version}")

    # check for existing model
    artifacts = os.listdir(f"./artifacts")

    # if version is not latest, check for existing version
    if version != "latest":
        if f"{model_name}:{version}" in artifacts:
            local_path = f"./artifacts/{model_name}:{version}"
            return local_path
    elif version == "latest":
        # get the latest model from wandb's api
        if f"{model_name}:{artifact._version}" in artifacts:
            local_path = f"./artifacts/{model_name}:{artifact._version}"
            return local_path

    return artifact.download()


if __name__ == "__main__":
    project_name = "text2midi-llm"
    model_name = "model-a0czmwoi"

    # project_name = "text2midi"
    # model_name = "model_best"

    version = "latest"
    local_path = get_model(project_name, model_name, version)
    print(local_path)
