import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def padded_cmap(solution, submission, padding_factor=5):
    """
    Kaggle's c-macro ROC AUC metric, optimized for evaluation.
    
    solution: ground truth (numpy array or tensor), shape [batch_size, num_classes]
    submission: predictions (numpy array or tensor), shape [batch_size, num_classes]
    padding_factor: default is 5.
    """
    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()
    if isinstance(submission, torch.Tensor):
        submission = submission.detach().cpu().numpy()
        
    new_rows = []
    
    # Check if there are any positive labels at all across the batch
    if solution.sum() == 0:
        return 0.5 # Default AUC for random guessing
        
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution[0]))])
        new_rows.append([0 for i in range(len(solution[0]))])
        
    solution = np.vstack([solution, new_rows])
    submission = np.vstack([submission, new_rows])
    
    score = roc_auc_score(
        solution, submission, average='macro'
    )
    
    return score

if __name__ == "__main__":
    print("=== Testando Métrica padded_cmap (ROC AUC macro) ===")
    dummy_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dummy_preds = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.3], [0.1, 0.2, 0.9]])
    
    score = padded_cmap(dummy_targets, dummy_preds)
    print(f"Dummy AUC Score: {score:.4f}")
