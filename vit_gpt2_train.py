
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import GPT2TokenizerFast, BertTokenizer, AutoFeatureExtractor, AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
import nltk
from PIL import Image
import time
import pandas as pd
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate


#########################################################################################################################
# In this code I am following a different method for training VIT-GPT model, using the links below:
# https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/vision-encoder-decoder
# https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/
#########################################################################################################################



# Load the pretrained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

####################### Added these lines for tokenization ##########################
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
#####################################################################################

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


output_dir = "vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


class ImageCapatioingDataset(torch.utils.data.Dataset):
    def __init__(self, ds, ds_type, max_target_length):
        self.ds = ds
        self.max_target_length = max_target_length
        self.ds_type = ds_type

    def __getitem__(self, idx):
        image_path = self.ds[self.ds_type]['image'][idx]
        caption = self.ds[self.ds_type]['caption'][idx]
        model_inputs = dict()
        model_inputs['labels'] = self.tokenization_fn(caption, self.max_target_length)
        model_inputs['pixel_values'] = self.feature_extraction_fn(image_path)
        return model_inputs

    def __len__(self):
        return len(self.ds[self.ds_type])
    
    # text preprocessing step
    def tokenization_fn(self, caption, max_target_length):
        """Run tokenization on caption."""
        labels = tokenizer(caption, 
                          padding="max_length", 
                          max_length=max_target_length).input_ids

        return labels
    
    # image preprocessing step
    def feature_extraction_fn(self, image_path):
        """
        Run feature extraction on images
        If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
        Otherwise, an exception will be thrown.
        """
        image = Image.open(image_path)

        encoder_inputs = feature_extractor(images=image, return_tensors="np")

        return encoder_inputs.pixel_values[0]

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                      padding="max_length", 
                      max_length=max_target_length).input_ids

    return labels

# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    model_inputs = {}

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image']
    captions = examples['caption']    
    
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs


ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result



if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training started")
    start_time = time.time()
    
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)
    
    
    
    # train_dataset = load_dataset("csv", data_files="img_cap_train_data_english.csv", delimiter=",")
    # val_dataset = load_dataset("csv", data_files="img_cap_val_data_english.csv", delimiter=",")
    # test_dataset = load_dataset("csv", data_files="img_cap_test_data_english.csv", delimiter=",")
    
    train_dataset = pd.read_csv("img_cap_train_data_english.csv").iloc[0: 30000]
    train_dataset.drop(['tokens', 'tokens_encoded'], axis=1, inplace=True)
    # Convert the DataFrame to a Dataset
    train_dataset = Dataset.from_pandas(train_dataset)

    val_dataset = pd.read_csv("img_cap_val_data_english.csv").iloc[0: 1000]
    val_dataset.drop(['tokens', 'tokens_encoded'], axis=1, inplace=True)
    # Convert the DataFrame to a Dataset
    val_dataset = Dataset.from_pandas(val_dataset)
    
    test_dataset = pd.read_csv("img_cap_test_data_english.csv").iloc[0: 10]
    test_dataset.drop(['tokens', 'tokens_encoded'], axis=1, inplace=True)
    # Convert the DataFrame to a Dataset
    test_dataset = Dataset.from_pandas(test_dataset)
    
    
    # Create the DatasetDict
    data_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
    })
    
    
    train_ds = ImageCapatioingDataset(data_dict, 'train', 64)
    val_ds = ImageCapatioingDataset(data_dict, 'validation', 64)
   
    
   
    # Define seq2seq training argumentsPermalink
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir="./image-captioning-output1",
    )
    
    
    # Define Metric
    metric = evaluate.load("rouge")
    
    
    # Let's Train!
    from transformers import default_data_collator

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    
    trainer.train()
    trainer.save_model("./image-captioning-output1")
    tokenizer.save_pretrained("./image-captioning-output1")
    # Save the fine-tuned model
    torch.save(model.state_dict(), "vit_gpt2_ft_revised_30000.pth")


