# Qwen3_4B_GRPO_Complete.py
# -------------------------------------------------------
# Complete GRPO training script for Qwen3-4B
# -------------------------------------------------------

import os
import torch
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported

# Environment setup
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # Extra 30% context lengths

# ============================================
# Configuration
# ============================================
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
MAX_TRAIN_SAMPLES = 100  # Adjust based on your needs

# Reasoning markers
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

# ============================================
# Model Setup
# ============================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,  # Use 16-bit for LoRA training
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.85,  # Reduced for stability
)

# Apply LoRA
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
SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '" + SYSTEM_PROMPT + "' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + REASONING_START + "' }}{% endif %}"
)
tokenizer.chat_template = chat_template

# Test chat template
print("\n" + "="*60)
print("Chat Template Test:")
print("="*60)
test_output = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": f"{REASONING_START}I think it's 2.{REASONING_END}{SOLUTION_START}2{SOLUTION_END}"},
        {"role": "user", "content": "What is 2+2?"},
    ],
    tokenize=False,
    add_generation_prompt=True
)
print(test_output)
print("="*60 + "\n")

# ============================================
# Dataset Loading and Preprocessing
# ============================================
print("Loading dataset...")
try:
    dataset = load_dataset("nvidia/OpenMathReasoning", split=f"train[:{MAX_TRAIN_SAMPLES}]")
    print(f"Loaded {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating dummy dataset for demonstration...")
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "problem": [
            "What is 5 + 3?",
            "Calculate 12 - 7",
            "What is 4 × 6?",
        ] * 10,
        "solution": ["8", "5", "24"] * 10,
    })

def format_dataset(examples):
    """Format dataset for GRPO training."""
    formatted = []
    
    for i in range(len(examples["problem"])):
        problem = examples["problem"][i]
        solution = examples["solution"][i] if "solution" in examples else "Unknown"
        
        # Create the conversation format
        conversation = [
            {"role": "user", "content": problem}
        ]
        
        formatted.append({
            "prompt": tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            ),
            "reference_solution": solution,
        })
    
    return {
        "prompt": [f["prompt"] for f in formatted],
        "reference_solution": [f["reference_solution"] for f in formatted],
    }

print("Formatting dataset...")
dataset = dataset.map(
    format_dataset,
    batched=True,
    remove_columns=dataset.column_names,
)

# ============================================
# Reward Function
# ============================================
def reward_function(completions, reference_solutions):
    """
    Calculate rewards based on how well the model's completion matches the reference.
    This is a simple exact match reward - you may want to make this more sophisticated.
    """
    rewards = []
    
    for completion, reference in zip(completions, reference_solutions):
        # Extract solution from completion
        if SOLUTION_START in completion and SOLUTION_END in completion:
            start_idx = completion.find(SOLUTION_START) + len(SOLUTION_START)
            end_idx = completion.find(SOLUTION_END)
            predicted_solution = completion[start_idx:end_idx].strip()
        else:
            predicted_solution = ""
        
        # Calculate reward (1.0 for exact match, 0.0 otherwise)
        # You can make this more sophisticated with partial credit
        reward = 1.0 if predicted_solution == reference.strip() else 0.0
        rewards.append(reward)
    
    return rewards

# ============================================
# Training Configuration
# ============================================
print("Setting up training configuration...")

training_args = GRPOConfig(
    output_dir="./qwen3_grpo_output",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    report_to="none",  # Change to "wandb" if you want logging
    # GRPO specific parameters
    num_sample_generations=2,  # Number of generations per prompt
    max_new_tokens=256,
)

# ============================================
# Trainer Setup
# ============================================
print("Initializing GRPO trainer...")

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_function=lambda completions: reward_function(
        completions, 
        dataset["reference_solution"]
    ),
)

# ============================================
# Training
# ============================================
print("\n" + "="*60)
print("Starting GRPO training...")
print("="*60 + "\n")

try:
    trainer.train()
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("This might be due to memory constraints or dataset issues.")

# ============================================
# Save Model
# ============================================
print("\nSaving model...")
model.save_pretrained("qwen3_grpo_lora")
tokenizer.save_pretrained("qwen3_grpo_lora")
print("✅ Model saved to './qwen3_grpo_lora'")

# ============================================
# Inference Function
# ============================================
def test_model(prompt, max_new_tokens=256):
    """Test the trained model with a prompt."""
    FastLanguageModel.for_inference(model)
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

# ============================================
# Test Inference
# ============================================
print("\n" + "="*60)
print("Testing model inference:")
print("="*60 + "\n")

test_prompts = [
    "What is 15 + 27?",
    "Calculate 8 × 9",
    "What is 100 - 43?"
]

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    response = test_model(prompt)
    print(f"Response: {response}\n")
    print("-" * 60 + "\n")

print("="*60)
print("✅ Script completed successfully!")
print("="*60)