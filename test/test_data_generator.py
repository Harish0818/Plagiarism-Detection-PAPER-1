import random
from typing import List, Dict
from faker import Faker

fake = Faker()

def generate_test_cases(num_cases: int = 100) -> List[Dict[str, any]]:
    """
    Generate test cases for plagiarism detection evaluation
    Returns list of dictionaries with 'text' and 'is_plagiarism' keys
    """
    test_cases = []
    
    # Generate some plagiarized content
    plagiarized_sources = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "A stitch in time saves nine."
    ]
    
    # Generate original content
    for _ in range(num_cases):
        is_plagiarism = random.random() < 0.3  # 30% plagiarized
        
        if is_plagiarism:
            # Create plagiarized text (either exact or paraphrased)
            source = random.choice(plagiarized_sources)
            if random.random() < 0.7:
                # Exact plagiarism
                text = source
            else:
                # Paraphrased plagiarism
                words = source.split()
                random.shuffle(words)
                text = ' '.join(words)
        else:
            # Generate original text
            text = fake.text(max_nb_chars=200)
        
        test_cases.append({
            'text': text,
            'is_plagiarism': int(is_plagiarism)
        })
    
    return test_cases