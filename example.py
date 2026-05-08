from ace import ACE
from typing import Any, Dict, List, Optional

# Initialize API clients
api_provider = "sambanova" # or "together", "openai", "commonstack"

# Initialize ACE system
ace_system = ACE(
    api_provider=api_provider,
    generator_model="DeepSeek-V3.1",
    reflector_model="DeepSeek-V3.1",
    curator_model="DeepSeek-V3.1",
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
    'test_workers': 20,
    'use_bulletpoint_analyzer': False,
    'api_provider': api_provider

}

train_samples: Optional[List[Dict[str, Any]]] = None,
val_samples: Optional[List[Dict[str, Any]]] = None,
test_samples: Optional[List[Dict[str, Any]]] = None,
data_processor: Optional[Any] = None  # Replace with your data processor if needed

# Offline adaptation
results = ace_system.run(
    mode='offline',
    train_samples=train_samples,
    val_samples=val_samples,
    test_samples=test_samples,  # Optional
    data_processor=data_processor, # Optional
    config=config
)

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