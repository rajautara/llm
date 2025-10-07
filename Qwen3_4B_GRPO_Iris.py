# Qwen3_4B_Iris_SFT.py
# -------------------------------------------------------
# Supervised Fine-tuning for Qwen3-4B on Iris Dataset
# (Fixed compatibility issues - using SFT instead of GRPO)
# -------------------------------------------------------

import os
import torch
import numpy as np
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel, is_bfloat16_supported

# Environment setup
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

# ============================================
# Configuration
# ============================================
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 3
TEST_SIZE = 0.2

# Reasoning markers
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

# Iris species mapping
SPECIES_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# ============================================
# Model Setup
# ============================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,  # Use 4-bit for better compatibility
    dtype=None,
    fast_inference=True,
)

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============================================
# Chat Template Setup
# ============================================
SYSTEM_PROMPT = f"""You are an expert botanist specializing in iris flower classification.
Given the measurements of an iris flower, analyze the features and classify it.
Think through your reasoning and place it between {REASONING_START} and {REASONING_END}.
Then, provide your final classification between {SOLUTION_START}{SOLUTION_END}.
The possible species are: Setosa, Versicolor, and Virginica."""

# Simplified chat template
chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""

tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# Load and Prepare Iris Dataset
# ============================================
print("\nLoading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(f"Dataset info:")
print(f"  - Total samples: {len(X)}")
print(f"  - Features: {feature_names}")
print(f"  - Classes: {list(SPECIES_NAMES.values())}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")

# ============================================
# Dataset Formatting
# ============================================
def create_iris_prompt(features, feature_names):
    """Create a natural language prompt from iris features."""
    sepal_length, sepal_width, petal_length, petal_width = features
    
    prompt = f"""Classify this iris flower based on its measurements:
- Sepal Length: {sepal_length:.1f} cm
- Sepal Width: {sepal_width:.1f} cm
- Petal Length: {petal_length:.1f} cm
- Petal Width: {petal_width:.1f} cm

What species is this iris flower?"""
    
    return prompt

def create_reasoning_response(features, species):
    """Create a response with reasoning for the classification."""
    sepal_length, sepal_width, petal_length, petal_width = features
    
    # Generate reasoning based on typical characteristics
    reasoning = []
    
    if species == "Setosa":
        reasoning.append(f"The petal length is {petal_length:.1f} cm, which is quite small.")
        reasoning.append(f"The petal width is {petal_width:.1f} cm, also small.")
        reasoning.append("These small petal measurements are characteristic of Setosa.")
    elif species == "Versicolor":
        reasoning.append(f"The petal length is {petal_length:.1f} cm, which is moderate.")
        reasoning.append(f"The sepal measurements are {sepal_length:.1f} cm x {sepal_width:.1f} cm.")
        reasoning.append("These moderate measurements suggest Versicolor.")
    else:  # Virginica
        reasoning.append(f"The petal length is {petal_length:.1f} cm, which is large.")
        reasoning.append(f"The petal width is {petal_width:.1f} cm, also large.")
        reasoning.append("These large petal measurements indicate Virginica.")
    
    reasoning_text = " ".join(reasoning)
    response = f"{REASONING_START}{reasoning_text}{REASONING_END}{SOLUTION_START}{species}{SOLUTION_END}"
    
    return response

def format_iris_dataset(X, y):
    """Format iris data for training with full conversations."""
    formatted_data = {
        "text": [],
        "reference_solution": [],
        "features": [],
    }
    
    for features, label in zip(X, y):
        species = SPECIES_NAMES[label]
        
        # Create full conversation
        prompt_text = create_iris_prompt(features, feature_names)
        response_text = create_reasoning_response(features, species)
        
        # Format as a complete conversation
        full_text = f"{SYSTEM_PROMPT}\n\n{prompt_text}\n{response_text}{tokenizer.eos_token}"
        
        formatted_data["text"].append(full_text)
        formatted_data["reference_solution"].append(species)
        formatted_data["features"].append(features.tolist())
    
    return formatted_data

print("\nFormatting training dataset...")
train_data = format_iris_dataset(X_train, y_train)
train_dataset = Dataset.from_dict(train_data)

print("Formatting test dataset...")
test_data = format_iris_dataset(X_test, y_test)
test_dataset = Dataset.from_dict(test_data)

# Show example
print("\n" + "="*60)
print("Example Training Sample:")
print("="*60)
print(train_dataset[0]["text"][:500] + "...")
print(f"\nReference Solution: {train_dataset[0]['reference_solution']}")
print("="*60 + "\n")

# ============================================
# Training Configuration
# ============================================
print("Setting up training configuration...")

training_args = TrainingArguments(
    output_dir="./qwen3_iris_output",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_steps=30,
    save_total_limit=2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_dir="./logs",
    report_to="none",
    max_grad_norm=0.3,
)

# ============================================
# Trainer Setup
# ============================================
print("Initializing SFT trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)

# ============================================
# Training
# ============================================
print("\n" + "="*60)
print("Starting Supervised Fine-tuning on Iris Dataset...")
print("="*60 + "\n")

try:
    trainer.train()
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# Save Model
# ============================================
print("\nSaving model...")
model.save_pretrained("qwen3_iris_lora")
tokenizer.save_pretrained("qwen3_iris_lora")
print("✅ Model saved to './qwen3_iris_lora'")

# ============================================
# Merge and Save Full Model (Optional)
# ============================================
print("\nMerging LoRA weights with base model...")
model.save_pretrained_merged("qwen3_iris_merged", tokenizer, save_method="merged_16bit")
print("✅ Merged model saved to './qwen3_iris_merged'")

# ============================================
# Evaluation Function
# ============================================
def evaluate_model(test_dataset):
    """Evaluate model on test set."""
    FastLanguageModel.for_inference(model)
    
    correct = 0
    total = len(test_dataset)
    results = []
    
    print("\n" + "="*60)
    print("Evaluating on Test Set...")
    print("="*60 + "\n")
    
    for i, sample in enumerate(test_dataset):
        features = sample["features"]
        reference = sample["reference_solution"]
        
        # Create prompt
        prompt_text = create_iris_prompt(features, feature_names)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}\n"
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract prediction
        if SOLUTION_START in response and SOLUTION_END in response:
            start_idx = response.find(SOLUTION_START) + len(SOLUTION_START)
            end_idx = response.find(SOLUTION_END)
            predicted = response[start_idx:end_idx].strip()
        else:
            predicted = "Unknown"
        
        is_correct = predicted.lower() == reference.lower()
        if is_correct:
            correct += 1
        
        results.append({
            "features": features,
            "reference": reference,
            "predicted": predicted,
            "correct": is_correct
        })
        
        if i < 5:  # Show first 5 predictions
            print(f"Sample {i+1}:")
            print(f"  Features: {features}")
            print(f"  Reference: {reference}")
            print(f"  Predicted: {predicted}")
            print(f"  Correct: {'✅' if is_correct else '❌'}")
            print()
    
    accuracy = (correct / total) * 100
    print("="*60)
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*60 + "\n")
    
    return results, accuracy

# ============================================
# Run Evaluation
# ============================================
results, accuracy = evaluate_model(test_dataset)

# ============================================
# Interactive Testing
# ============================================
def test_custom_iris(sepal_length, sepal_width, petal_length, petal_width):
    """Test model with custom iris measurements."""
    FastLanguageModel.for_inference(model)
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prompt_text = create_iris_prompt(features, feature_names)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test with example measurements
print("\n" + "="*60)
print("Testing with Custom Measurements:")
print("="*60 + "\n")

test_cases = [
    (5.1, 3.5, 1.4, 0.2),  # Typical Setosa
    (6.5, 2.8, 4.6, 1.5),  # Typical Versicolor
    (7.2, 3.0, 5.8, 1.6),  # Typical Virginica
]

for i, (sl, sw, pl, pw) in enumerate(test_cases, 1):
    print(f"Test Case {i}: Sepal({sl}, {sw}), Petal({pl}, {pw})")
    response = test_custom_iris(sl, sw, pl, pw)
    print(f"Response:\n{response}\n")
    print("-" * 60 + "\n")

print("="*60)
print("✅ Script completed successfully!")
print(f"Final Test Accuracy: {accuracy:.2f}%")
print("="*60)