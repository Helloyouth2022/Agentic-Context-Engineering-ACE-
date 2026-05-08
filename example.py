from ace import ACE
from typing import Any, Dict, List, Optional

# Initialize API clients
api_provider = "dashscope" # or "together", "openai", "commonstack", "dashscope"
model_name = "deepseek-v3.1"  # or any other model supported by your API provider

# Initialize ACE system
ace_system = ACE(
    api_provider=api_provider,
    generator_model=model_name,
    reflector_model=model_name,
    curator_model=model_name,
    max_tokens=4096
)

# Prepare configuration
config = {
    'num_epochs': 1,
    'max_num_rounds': 3,
    'curator_frequency': 1,
    'eval_steps': 100,
    'online_eval_frequency': 15,
    'save_steps': 50,
    'playbook_token_budget': 80000,
    'task_name': 'your_task',
    'json_mode': False,
    'no_ground_truth': False,
    'save_dir': './results',
    'test_workers': 1,
    'use_bulletpoint_analyzer': False,
    'api_provider': api_provider

}

train_samples: Optional[List[Dict[str, Any]]] = [{"context": "test", "question": "test", "target": "test", "others": {...}}]
val_samples: Optional[List[Dict[str, Any]]] = [{"context": "test", "question": "test", "target": "test", "others": {...}}]
test_samples: Optional[List[Dict[str, Any]]] = [
    {"context": "Urban gardening has become increasingly popular in recent years, especially among young adults living in cities. With limited access to outdoor space, many city dwellers have turned to creative solutions such as rooftop gardens, balcony planters, and community garden plots. Not only does urban gardening provide fresh produce, but it also offers mental health benefits by connecting people with nature. Some cities even offer subsidies or workshops to encourage residents to start their own small gardens. Despite challenges like poor soil quality and lack of sunlight in densely built areas, enthusiasts continue to find ways to grow herbs, vegetables, and flowers in unexpected places.", 
     "question": """What is one reason urban gardening has gained popularity among young adults in cities?
A) It allows them to avoid paying for groceries entirely.
B) It provides both fresh food and mental health benefits.
C) It guarantees access to large outdoor spaces.
D) It eliminates the need for community interaction. """, 
     "target": "B"}]


class DataProcessor:
    def process_task_data(self, raw_data):
        # Convert your data format to standardized format
        return [{"context": ..., "question": ..., "target": ..., "others": {...}}]
    
    def answer_is_correct(self, predicted, ground_truth):
        # Your comparison logic
        predicted, ground_truth = predicted.strip(), ground_truth.strip()
        predicted = predicted[:len(ground_truth)]
        return predicted == ground_truth
    
    def evaluate_accuracy(self, predictions, ground_truths):
        # Calculate accuracy
        return sum(self.answer_is_correct(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)
data_processor: Optional[Any] = DataProcessor()  # Replace with your data processor if needed

# Offline adaptation
# results = ace_system.run(
#     mode='offline',
#     train_samples=train_samples,
#     val_samples=val_samples,
#     test_samples=None,  # Optional
#     data_processor=data_processor, # Optional
#     config=config
# )

# Online adaptation
results = ace_system.run(
    mode='online',
    test_samples=test_samples,
    data_processor=data_processor,
    config=config
)

# Evaluation only
results = ace_system.run(
    mode='eval_only',
    test_samples=test_samples,
    data_processor=data_processor,
    config=config
)