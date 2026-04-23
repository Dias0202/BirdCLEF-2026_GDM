import mlflow
import os

def test_mlflow():
    print("=== Phase 1: MLflow Initialization and Test ===")
    
    # Configure MLflow directory to be within the project folder
    mlflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
    
    # Define experiment name
    experiment_name = "BirdCLEF_2026_EDA_Setup"
    mlflow.set_experiment(experiment_name)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    
    # Start a run and log dummy metrics
    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        
        # Log parameters (e.g., hyperparams from config)
        mlflow.log_param("BATCH_SIZE", 16)
        mlflow.log_param("CHUNK_DURATION", 5)
        
        # Log metrics (e.g., dummy loss)
        for epoch in range(3):
            mlflow.log_metric("dummy_train_loss", 0.5 - (epoch * 0.1), step=epoch)
            mlflow.log_metric("dummy_val_auc", 0.6 + (epoch * 0.1), step=epoch)
            
        print("-> Dummy parameters and metrics logged successfully.")
        
    print("-> MLflow test completed successfully. You can visualize by running: `mlflow ui` in the terminal.")

if __name__ == "__main__":
    test_mlflow()