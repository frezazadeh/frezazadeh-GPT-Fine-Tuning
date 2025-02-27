import pickle
from openai import OpenAI
from huggingface_hub import login
from src import config, utils

# Log in to HuggingFace using the token from the configuration
login(config.HF_TOKEN, add_to_git_credential=True)

openai = OpenAI(api_key=config.OPENAI_API_KEY)

def load_data(train_path: str, test_path: str):
    """Load train and test datasets from pickle files."""
    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    with open(test_path, 'rb') as f:
        test = pickle.load(f)
    return train, test

def prepare_datasets(train, validation_ratio=0.25, train_count=200):
    """Split the training data into fine-tuning training and validation sets."""
    fine_tune_train = train[:train_count]
    fine_tune_validation = train[train_count:train_count+int(len(train)*validation_ratio)]
    return fine_tune_train, fine_tune_validation

def write_jsonl(items, filename: str):
    """Write items into a JSONL file for fine-tuning."""
    jsonl_content = utils.make_jsonl(items)
    with open(filename, "w") as f:
        f.write(jsonl_content)

def upload_files(train_filename: str, validation_filename: str):
    """Upload JSONL files to OpenAI for fine-tuning."""
    with open(train_filename, "rb") as f:
        train_file = openai.files.create(file=f, purpose="fine-tune")
    with open(validation_filename, "rb") as f:
        validation_file = openai.files.create(file=f, purpose="fine-tune")
    return train_file, validation_file

def create_fine_tuning_job(train_file_id: str, validation_file_id: str, wandb_project=None):
    """
    Create and start a fine-tuning job.
    If wandb_project is provided, the WandB integration is enabled.
    """
    integrations = []
    if wandb_project:
        integrations.append({"type": "wandb", "wandb": {"project": wandb_project}})
    
    job = openai.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=validation_file_id,
        model="gpt-4o-mini-2024-07-18",
        seed=42,
        hyperparameters={"n_epochs": 1},
        integrations=integrations,
        suffix="pricer"
    )
    return job

