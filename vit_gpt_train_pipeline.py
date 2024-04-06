import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import NLLLoss
# from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
# I am using ViTImageProcessor instead of ViTFeatureExtractor since its deprecated
from transformers import (VisionEncoderDecoderModel, 
                          AutoTokenizer,ViTImageProcessor)
# The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.
import pandas as pd
from PIL import Image
import os 
import ast
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
import time



# Note that there is alos a dataloader in this file but I am using pytorchs dataloader
from data import (ArtEmis, ArtEmisDetectionsField, EmotionField, LanguageField,
                  RawField, TextField, HuggingfaceVocab)


from evaluation import Cider, PTBTokenizer

################
# Run with this command 
# CUDA_LAUNCH_BLOCKING=1 python3 vit_gpt_clean.py
#############


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define your dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, image_processor):
        self.data = pd.read_csv(csv_file).iloc[0: 1000]
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
        encoded_caption = self.tokenizer.encode_plus(caption, padding="max_length", truncation=True, max_length=46,
                                                     return_tensors="pt")
        
        labels = self.tokenizer(caption, padding="max_length", truncation=True, max_length=46, return_tensors="pt",).input_ids

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
        
        

# def train_epoch(model, data_loader, loss_function, optimizer):
#     for i, batch in enumerate(data_loader):
#         encoded_caption = batch['encoded_caption'].to(device)
#         # print(type(batch['tokens_encoded']))
#         processed_img_pixel_values = batch['processed_img_pixel_values'].to(device)

#         # Forward pass
#         outputs = model(pixel_values=processed_img_pixel_values, decoder_input_ids=encoded_caption)
#         logits = outputs.logits

#         # Compute the loss
#         loss = loss_function(logits.view(-1, logits.size(-1)), encoded_caption.view(-1))

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         return loss


def train_epoch(model, data_loader, loss_function, optimizer, batch_size, toknization_method):
    for i, batch in enumerate(data_loader):

        captions_gt = batch['tokens_encoded'].to(device)
        if toknization_method == "load_from_dataset":
            encoded_caption = batch['tokens_encoded'].to(device)

            
        elif toknization_method == "tokenize_with_our_model":
            
            encoded_caption = batch['encoded_caption'].to(device)

            
        processed_img_pixel_values = batch['processed_img_pixel_values'].to(device)
        labels = batch['labels'].to(device)
        loss = model(pixel_values=processed_img_pixel_values, labels=labels).loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss



if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training started")
    start_time = time.time()
    
    
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    
    # Load the pretrained model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    ####################### Added these lines for tokenization ##########################
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    #####################################################################################
    
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    
    # Freeze the pretrained layers
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Make only the parameters of the final layers trainable
    # final_layers = model.decoder.parameters()  # Customize this based on your model's architecture
    # final_layers = model.decoder.lm_head.parameters()
    # final_layers = model.decoder.parameters()
    # for param in final_layers:
    #     param.requires_grad = True

    # Define the fine-tuning parameters
    lr = 1e-5
    batch_size = 8
    num_epochs = 2
    
    # Load your dataset
    dataset = ImageCaptionDataset("img_cap_train_data_english.csv", tokenizer, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set the number of GPUs you want to use
    num_gpus = 4

    # Use this line to make sure the data is loaded on the correct GPU
    torch.cuda.set_device(0)
    
    
    # Wrap your model with DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Move the model to the device
    model = model.to(device)


    # Define the loss function
    loss_function = nn.CrossEntropyLoss()    
    # print(text_field)
    # loss_function = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    # loss_function = NLLLoss()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training without emotion grounding
    for e in range(num_epochs):
        acc_loss = 0
        for i, batch in enumerate(data_loader):
                
            processed_img_pixel_values = batch['processed_img_pixel_values'].to(device)
            print(processed_img_pixel_values.shape)
            labels = batch['labels'].to(device)
            print(labels.shape)
            loss = model(pixel_values=processed_img_pixel_values, labels=labels).loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_loss = acc_loss + loss.item()
            
        print(acc_loss)
    
    print("--- %s Training finished. It took seconds ---" % (time.time() - start_time))
    # Save the fine-tuned model
    torch.save(model.state_dict(), "vit_gpt2_ft_my_tokens_50000_10_epochs_lr_1e6_only_decoder.pth")
    
    
    