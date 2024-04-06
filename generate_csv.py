import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoProcessor
from datasets import load_dataset
import os
import ast


df = pd.read_csv("artelingo_release_valid_paths.csv", encoding='utf-8')

language = "chinese"

# Prepare the data in the required format
train_data = {
    'image': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['image_file'].tolist(),
    'caption': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['utterance'].tolist(),
    'emotion': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion'].tolist(),
    'emotion_label': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion_label'].tolist(),
    'language': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['language'].tolist(),
    'tokens': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens'].tolist(),
    'tokens_encoded': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_encoded'].tolist(),
    'tokens_len': df[(df["split"]=="train") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_len'].tolist()
}

val_data = {
    'image': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['image_file'].tolist(),
    'caption': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['utterance'].tolist(),
    'emotion': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion'].tolist(),
    'emotion_label': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion_label'].tolist(),
    'language': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['language'].tolist(),
    'tokens': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens'].tolist(),
    'tokens_encoded': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_encoded'].tolist(),
    'tokens_len': df[(df["split"]=="val") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_len'].tolist()
}

test_data = {
    'image': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['image_file'].tolist(),
    'caption': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['utterance'].tolist(),
    'emotion': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion'].tolist(),
    'emotion_label': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['emotion_label'].tolist(),
    'language': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['language'].tolist(),
    'tokens': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens'].tolist(),
    'tokens_encoded': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_encoded'].tolist(),
    'tokens_len': df[(df["split"]=="test") & (df["language"]==f"{language}") & (df["is_valid"]==True)]['tokens_len'].tolist()
}

# Convert the data to a dataset
train_dataset = pd.DataFrame(train_data)
val_dataset = pd.DataFrame(val_data)
test_dataset = pd.DataFrame(test_data)



# Save the dataset as a CSV file
train_dataset.to_csv(f"img_cap_train_data_{language}.csv", index=False, encoding='utf-8')  
val_dataset.to_csv(f"img_cap_val_data_{language}.csv", index=False, encoding='utf-8')  
test_dataset.to_csv(f"img_cap_test_data_{language}.csv", index=False, encoding='utf-8')