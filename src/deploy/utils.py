import os
import wandb
import argparse
from dotenv import load_dotenv, find_dotenv


ARTIFACTS_DIR = "./artifacts"


def wandb_login():
    """login to wandb -- wraps around wandb.login() with env file loading"""
    load_dotenv(find_dotenv())
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_model(
    project_name: str,
    model_name: str,
    version: str = "latest",
    cleanup: bool = True,
):
    # login to wandb
    wandb_login()

    # check for existing model
    artifacts = os.listdir(ARTIFACTS_DIR) if os.path.exists(ARTIFACTS_DIR) else []
    artifacts = [x for x in artifacts if x.startswith(f"{project_name}/{model_name}")]

    # get the latest model from wandb's api
    api = wandb.Api()
    artifact = api.artifact(f"{project_name}/{model_name}:{version}")

    # get the version from the artifact, does the path exist
    # we do this bc latest is technically not a version
    if artifact._version is not None:
        # compute the path where it could be
        possible_path = f"{project_name}/{model_name}:{artifact._version}"

        # check if it exists
        if possible_path in artifacts:
            return f"{ARTIFACTS_DIR}/{possible_path}"

        # if it doesn't exist, we need to download it
        file_path = artifact.download(ARTIFACTS_DIR)

        # check if we need to cleanup any old models
        if cleanup:
            for artifact in artifacts:
                os.remove(f"{ARTIFACTS_DIR}/{artifact}")

        # return the path
        return file_path

    else:
        # the model was not found in wandb, throw an error
        raise ValueError(f"Model {model_name} not found in wandb")


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
