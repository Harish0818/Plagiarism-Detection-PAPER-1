from utils.plagiarism import PlagiarismDetector
from utils.evaluation import evaluate_plagiarism_detector
from tests.test_data_generator import generate_test_cases
import pandas as pd

def run_evaluation():
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Generate test cases (in a real system, you'd use a labeled dataset)
    test_cases = generate_test_cases(1000)
    
    # Run evaluation
    metrics = evaluate_plagiarism_detector(detector, test_cases, threshold=0.8)
    
    # Display results
    print("\nPlagiarism Detection Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    cm_df = pd.DataFrame({
        '': ['Actual Positive', 'Actual Negative'],
        'Predicted Positive': [cm['true_positives'], cm['false_positives']],
        'Predicted Negative': [cm['false_negatives'], cm['true_negatives']]
    }).set_index('')
    print(cm_df)

if __name__ == "__main__":
    run_evaluation()