import time
import src.config as config
import src.fine_tune as fine_tune
import src.utils as utils
import src.testing as testing
import src.items as items
from openai import OpenAI

def poll_job_events(job_id, sleep_time=300, max_retries=30):
    """
    Polls the fine-tuning job for events and prints the current job status.
    Returns the fine-tuned model name once complete.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    for attempt in range(max_retries):
        # Retrieve current job status
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Polling events (attempt {attempt + 1}):")
        print("Job status:", job_status.status)  # Print overall status
        
        # Retrieve and print job events
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id).data
        for event in events:
            print(event)
        
        # Check if the fine-tuned model is available (job complete)
        if job_status.fine_tuned_model:
            print("Fine-tuning complete!")
            return job_status.fine_tuned_model
        
        time.sleep(sleep_time)
    
    raise RuntimeError("Fine tuning job did not complete in the expected time.")

def main():
    # Load datasets from the data directory
    train, test = fine_tune.load_data("data/processed/train.pkl", "data/processed/test.pkl")
    
    # Prepare fine-tuning datasets
    fine_tune_train, fine_tune_validation = fine_tune.prepare_datasets(train)
    
    # Write JSONL files for fine-tuning
    fine_tune.write_jsonl(fine_tune_train, "data/processed/fine_tune_train.jsonl")
    fine_tune.write_jsonl(fine_tune_validation, "data/processed/fine_tune_validation.jsonl")
    
    # Upload files to OpenAI
    train_file, validation_file = fine_tune.upload_files(
        "data/processed/fine_tune_train.jsonl",
        "data/processed/fine_tune_validation.jsonl"
    )
    
    # Create a fine-tuning job
    job = fine_tune.create_fine_tuning_job(train_file.id, validation_file.id)
    # Or, if you have configured WandB in your organization:
    # job = fine_tune.create_fine_tuning_job(train_file.id, validation_file.id, wandb_project="gpt-pricer")
    
    # Poll for job events and wait for the fine-tuning job to complete
    fine_tuned_model_name = poll_job_events(job.id)
    print("Fine tuned model name:", fine_tuned_model_name)
    
    # Define a prediction function using the fine-tuned model
    def predict(item_instance):
        response = OpenAI(api_key=config.OPENAI_API_KEY).chat.completions.create(
            model=fine_tuned_model_name, 
            messages=utils.messages_for(item_instance, include_price=False),
            seed=42,
            max_tokens=7
        )
        reply = response.choices[0].message.content
        return utils.get_price(reply)

    # Run tests on the fine-tuned model using the Tester class
    testing.Tester.test(predict, test)

if __name__ == "__main__":
    main()