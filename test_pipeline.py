import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import NLLLoss
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import pandas as pd
from PIL import Image
import os 
import ast
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
import time
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider


import matplotlib.pyplot as plt
from transformers import pipeline

# Note that there is alos a dataloader in this file but I am using pytorchs dataloader
# from data import (ArtEmis, ArtEmisDetectionsField, EmotionField, LanguageField,
#                   RawField, TextField, HuggingfaceVocab)




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define your dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, image_processor):
        self.data = pd.read_csv(csv_file).iloc[0: 10000]
        # self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path      = self.data.iloc[idx]['image']
        caption         = self.data.iloc[idx]['caption']
        emotion         = self.data.iloc[idx]['emotion']
        emotion_label   = self.data.iloc[idx]['emotion_label']
        language        = self.data.iloc[idx]['language']
        tokens          = self.data.iloc[idx]['tokens']
        tokens_encoded  = self.data.iloc[idx]['tokens_encoded']
        tokens_len      = self.data.iloc[idx]['tokens_len']

        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        encoded_caption = self.tokenizer.encode_plus(caption, padding="max_length", truncation=True, max_length=64,
                                                     return_tensors="pt")
        
        labels = self.tokenizer(caption, padding="max_length", truncation=True, max_length=64, return_tensors="pt",).input_ids

        return {
            'encoded_caption': encoded_caption['input_ids'].squeeze(),
            'encoded_caption_attention_mask': encoded_caption['attention_mask'].squeeze(),
            'processed_img_pixel_values': image.pixel_values.squeeze(),
            'emotion': emotion,
            'emotion_label': emotion_label,
            'language': language,
            'tokens': tokens,
            'tokens_encoded': torch.tensor(ast.literal_eval(tokens_encoded)[0:46]),
            'tokens_len': tokens_len,
            'labels': labels
        }
        
        

def generate_outputs(model, tokenizer, image_processor, image_paths):
    
    all_generated_captions = []
    for image_path in image_paths:
    
        image = Image.open(image_path)
        image = image_processor(image, return_tensors="pt")
        pixel_values = image.pixel_values.to(device)

        # Generate the caption
        outputs = model.generate(pixel_values=pixel_values)
        # predicted_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_caption = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print("Predicted Caption:", predicted_caption)
        all_generated_captions.append([predicted_caption])

    return all_generated_captions

    

def plot_img_with_captions(image, reference_caption, generated_caption):
    
    # Create a 1x2 subplot and plot images with labels
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figure size if needed

    # Plot the first image with label
    axes[0].imshow(image)
    axes[0].set_title(f"Reference Caption: {reference_caption[0]}")
    axes[0].axis('off')

    # Plot the second image with label
    axes[1].imshow(image)
    axes[1].set_title(f"Generated Caption: {generated_caption[0]}")
    axes[1].axis('off')
    plt.show()
    

if __name__ == "__main__":
    
    nltk.download('wordnet')
    
    # torch.cuda.empty_cache()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Load the saved model
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    model.load_state_dict(torch.load("/home/benyamin/Desktop/ArtElingo/vit_gpt2_ft_revised_30000.pth"))
    ####################### Added these lines for tokenization ##########################
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    
    # Load your dataset
    test_dataset = ImageCaptionDataset("img_cap_test_data_english.csv", tokenizer, image_processor)
    all_image_paths = test_dataset.data.image.tolist()
    all_reference_captions = test_dataset.data.caption.tolist()
    
    # Let's generate captions usning the fine-tuned model
    # all_generated_captions = generate_outputs(model, tokenizer, image_processor, all_image_paths[0: 50])
    all_generated_captions = generate_outputs(model, tokenizer, image_processor, all_image_paths)

    
    # Initialize scorers
    cider_scorer = Cider()

    # Lists to store individual scores
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    cider_scores = []

    
    smooth = SmoothingFunction().method1
    
    # Loop through all data points
    for data_id in range(1, len(all_generated_captions)):
        image = Image.open(all_image_paths[data_id])
    
        reference_caption = [all_reference_captions[data_id]]
        generated_caption = all_generated_captions[data_id]
        print("reference caption: ",reference_caption)
        print("generated caption: ", generated_caption)
        
        plot_img_with_captions(image, reference_caption, generated_caption)

        # Calculate BLEU score
        bleu_score = corpus_bleu([reference_caption], generated_caption, smoothing_function=smooth)
        print("BLEU Score is: ", bleu_score)
        bleu_scores.append(bleu_score)

        # Calculate METEOR score
        Meteor_score = meteor_score([reference_caption], generated_caption)
        print("Meteor Score is: ", Meteor_score)
        meteor_scores.append(Meteor_score)

        # Calculate ROUGE score (using BLEU-like approach for simplicity)
        rouge_score = sentence_bleu([caption.split() for caption in reference_caption], generated_caption[0].split())
        print("Rouge Score is: ", rouge_score)
        rouge_scores.append(rouge_score)

        # Calculate CIDEr score
        cider_score, _ = cider_scorer.compute_score({data_id: reference_caption}, {data_id: generated_caption})
        print("CIDER Score is: ", cider_score)
        cider_scores.append(cider_score)

    # Calculate the average scores
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    average_rouge = sum(rouge_scores) / len(rouge_scores)
    average_cider = sum(cider_scores) / len(cider_scores)

    # Print or store the average scores
    print("Average BLEU Score:", average_bleu)
    print("Average METEOR Score:", average_meteor)
    print("Average ROUGE Score:", average_rouge)
    print("Average CIDEr Score:", average_cider)

    