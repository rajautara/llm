#!/usr/bin/env python
# coding: utf-8

# GRPO Training for Iris Classification with Reasoning

# Installation (run in Colab)
# %%capture
# import os
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
# !pip install --upgrade -qqq uv
# !uv pip install -qqq --upgrade unsloth vllm
# !uv pip install transformers==4.55.4
# !uv pip install --no-deps trl==0.22.2

from unsloth import FastLanguageModel
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
import re

# ============================================================================
# STEP 1: Load Model
# ============================================================================
max_seq_length = 1024  # Shorter for classification tasks
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ============================================================================
# STEP 2: Define Custom Tags and System Prompt
# ============================================================================
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = f"""You are a botanist analyzing iris flowers.
Given measurements of an iris flower, analyze the features carefully.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide the species name between {solution_start}{solution_end}
The species must be one of: setosa, versicolor, or virginica"""

# ============================================================================
# STEP 3: Create Chat Template
# ============================================================================
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# ============================================================================
# STEP 4: Load and Prepare Iris Dataset
# ============================================================================
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
iris_df = pd.DataFrame(
    data=iris.data,
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({
    0: 'setosa',
    1: 'versicolor', 
    2: 'virginica'
})

print(f"Iris dataset shape: {iris_df.shape}")
print(iris_df.head())

# ============================================================================
# STEP 5: Create Reasoning Examples for Pre-training
# ============================================================================
def create_reasoning_example(row):
    """Create a reasoning example for each flower"""
    species = row['species_name']
    
    # Create different reasoning styles
    reasoning_templates = [
        f"Looking at the measurements:\n"
        f"- Sepal length: {row['sepal_length']} cm\n"
        f"- Sepal width: {row['sepal_width']} cm\n"
        f"- Petal length: {row['petal_length']} cm\n"
        f"- Petal width: {row['petal_width']} cm\n\n"
        f"The petal length of {row['petal_length']} cm is {'very short' if row['petal_length'] < 2 else 'moderate' if row['petal_length'] < 5 else 'long'}. "
        f"The petal width of {row['petal_width']} cm is {'narrow' if row['petal_width'] < 1 else 'medium' if row['petal_width'] < 1.8 else 'wide'}. "
        f"Based on these characteristics, this appears to be {species}.",
        
        f"Analyzing the flower:\n"
        f"Sepal: {row['sepal_length']}cm × {row['sepal_width']}cm\n"
        f"Petal: {row['petal_length']}cm × {row['petal_width']}cm\n\n"
        f"{'Setosa typically has very small petals (< 2cm).' if species == 'setosa' else ''}"
        f"{'Versicolor has medium-sized petals (3-5cm).' if species == 'versicolor' else ''}"
        f"{'Virginica has large petals (> 5cm).' if species == 'virginica' else ''}"
        f" This matches {species}.",
    ]
    
    reasoning = np.random.choice(reasoning_templates)
    return reasoning

# Create pre-training dataset (use 30 samples)
pretrain_df = iris_df.sample(n=30, random_state=42).copy()
pretrain_df['reasoning'] = pretrain_df.apply(create_reasoning_example, axis=1)

def format_pretrain_dataset(row):
    problem = f"""Classify this iris flower:
Sepal Length: {row['sepal_length']} cm
Sepal Width: {row['sepal_width']} cm
Petal Length: {row['petal_length']} cm
Petal Width: {row['petal_width']} cm"""
    
    final_prompt = (
        reasoning_start + row['reasoning'] + reasoning_end +
        solution_start + row['species_name'] + solution_end
    )
    
    return {
        "Messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]
    }

pretrain_dataset = pretrain_df.apply(format_pretrain_dataset, axis=1).tolist()
pretrain_dataset = pd.DataFrame(pretrain_dataset)

# Tokenize
pretrain_dataset["text"] = tokenizer.apply_chat_template(
    pretrain_dataset["Messages"].values.tolist(), 
    tokenize=False
)
pretrain_dataset = Dataset.from_pandas(pretrain_dataset)

print(f"\nPre-training dataset size: {len(pretrain_dataset)}")
print("\nExample formatted message:")
print(tokenizer.apply_chat_template(pretrain_dataset[0]["Messages"], tokenize=False))

# ============================================================================
# STEP 6: Pre-Fine-Tune for Format
# ============================================================================
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=pretrain_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

print("\n" + "="*50)
print("Starting pre-training...")
print("="*50)
trainer.train()

# Test the model
text = tokenizer.apply_chat_template(
    pretrain_dataset[0]["Messages"][:2],
    tokenize=False,
    add_generation_prompt=True,
)

from transformers import TextStreamer
print("\n" + "="*50)
print("Testing pre-trained model:")
print("="*50)
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=0,
    max_new_tokens=512,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)

# Clean up
del pretrain_dataset
torch.cuda.empty_cache()
import gc
gc.collect()

# ============================================================================
# STEP 7: Prepare GRPO Dataset
# ============================================================================
# Use remaining samples for GRPO training
grpo_df = iris_df[~iris_df.index.isin(pretrain_df.index)].copy()

def format_grpo_dataset(row):
    problem = f"""Classify this iris flower:
Sepal Length: {row['sepal_length']} cm
Sepal Width: {row['sepal_width']} cm
Petal Length: {row['petal_length']} cm
Petal Width: {row['petal_width']} cm"""
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        "answer": row['species_name']
    }

grpo_dataset = grpo_df.apply(format_grpo_dataset, axis=1).tolist()
grpo_dataset = pd.DataFrame(grpo_dataset)
grpo_dataset = Dataset.from_pandas(grpo_dataset)

print(f"\nGRPO dataset size: {len(grpo_dataset)}")
print(f"First example: {grpo_dataset[0]}")

# ============================================================================
# STEP 8: Define Reward Functions
# ============================================================================
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    """Reward correct formatting"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 3.0 if match_format.search(response) is not None else 0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    """Reward partial formatting"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """Reward correct species classification"""
    responses = [completion[0]["content"] for completion in completions]
    
    extracted_responses = [
        guess.group(1).strip().lower()
        if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.0)
            continue
        
        true_answer = true_answer.strip().lower()
        
        # Exact match gets high reward
        if guess == true_answer:
            scores.append(5.0)
        # Partial match (contains the right species)
        elif true_answer in guess or guess in true_answer:
            scores.append(3.0)
        else:
            scores.append(-3.0)
    
    return scores

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def print_examples(prompts, completions, answer, **kwargs):
    """Print examples periodically"""
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"]
        response = completions[0][0]["content"]
        
        extracted = None
        if (match := match_format.search(response)):
            extracted = match.group(1).strip()
        
        print("\n" + "="*60)
        print(f"STEP {PRINTED_TIMES}")
        print("="*60)
        print(f"Question:\n{question}")
        print(f"\nTrue Answer: {answer[0]}")
        print(f"\nModel Response:\n{response}")
        print(f"\nExtracted Answer: {extracted}")
        print("="*60)
    
    PRINTED_TIMES += 1
    return [0] * len(completions)  # Neutral reward

# ============================================================================
# STEP 9: GRPO Training
# ============================================================================
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

max_prompt_length = 512
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=2,
    save_steps=50,
    output_dir="outputs",
    report_to="none",
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,  # FIXED: Changed from 'config' to 'args'
    train_dataset=grpo_dataset,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        print_examples,
    ],
)

print("\n" + "="*50)
print("Starting GRPO training...")
print("="*50)
grpo_trainer.train()

# ============================================================================
# STEP 10: Test the Model
# ============================================================================
print("\n" + "="*50)
print("Testing final model on new examples:")
print("="*50)

test_examples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
    {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
]

for i, example in enumerate(test_examples):
    problem = f"""Classify this iris flower:
Sepal Length: {example['sepal_length']} cm
Sepal Width: {example['sepal_width']} cm
Petal Length: {example['petal_length']} cm
Petal Width: {example['petal_width']} cm"""
    
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Test Example {i+1}")
    print('='*60)
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=0.3,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

# ============================================================================
# STEP 11: Save the Model
# ============================================================================
model.save_pretrained("iris_reasoning_model")
tokenizer.save_pretrained("iris_reasoning_model")
print("\n✓ Model saved to 'iris_reasoning_model'")
