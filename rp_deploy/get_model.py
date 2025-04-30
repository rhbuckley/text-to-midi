import os
import wandb
from dotenv import load_dotenv, find_dotenv
import argparse


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
    artifacts = os.listdir(f"./artifacts") if os.path.exists("./artifacts") else []

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
    parser = argparse.ArgumentParser(
        description="Download a model artifact from WandB."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the WandB project.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model artifact."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Version of the model artifact (default: latest).",
    )

    args = parser.parse_args()

    local_path = get_model(args.project_name, args.model_name, args.version)
    print(f"Model downloaded to: {local_path}")
