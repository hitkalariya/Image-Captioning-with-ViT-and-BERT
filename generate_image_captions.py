"""
Image Captioning Inference Script
--------------------------------
This script loads a trained Vision Transformer + BERT model and generates captions for input images.

Author: Hit Kalariya
GitHub: https://github.com/hitkalariya
"""
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
import sys

def load_image_captioning_model(model_dir):
    """Load the saved image captioning model and its components.
    Args:
        model_dir (str): Path to the model directory or .safetensors file.
    Returns:
        model, decoder_tokenizer, image_processor, device
    """
    # Load the model - use safetensors if available
    try:
        # Try loading the model directly from the directory
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    except:
        # If that fails, try loading using the model file structure
        model_path = model_dir
        if model_path.endswith('.safetensors'):
            # Get the directory containing the .safetensors file
            import os
            model_dir = os.path.dirname(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Load the tokenizer and image processor
    decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Ensure pad token is set
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, decoder_tokenizer, image_processor, device

def generate_caption(image_path, model, decoder_tokenizer, image_processor, device, show_image=True):
    """Generate a caption for an input image.
    Args:
        image_path (str): Path to the image file.
        model: Loaded VisionEncoderDecoderModel.
        decoder_tokenizer: Tokenizer for the decoder.
        image_processor: Image processor for the encoder.
        device: torch.device.
        show_image (bool): Whether to display the image.
    Returns:
        str: Generated caption.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"[ERROR] Could not open image: {e}")
        return None
    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    image = image.resize((224, 224))
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=decoder_tokenizer.cls_token_id,
            max_length=32
        )
    caption = decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.replace(".", "")
    print(f"Generated Caption: {caption}")
    print("(Model by Hit Kalariya - https://github.com/hitkalariya)")
    return caption

# Example CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate an image caption using a ViT-BERT model.")
    parser.add_argument('--model', type=str, default="/content/model.safetensors", help='Path to model directory or .safetensors file')
    parser.add_argument('--image', type=str, default="/content/flickr8k/Images/2757779501_c41c86a595.jpg", help='Path to the image file')
    parser.add_argument('--no-show', action='store_true', help='Do not display the image')
    args = parser.parse_args()
    model, decoder_tokenizer, image_processor, device = load_image_captioning_model(args.model)
    caption = generate_caption(args.image, model, decoder_tokenizer, image_processor, device, show_image=not args.no_show)
    if caption:
        print(f"\n[INFO] Caption: {caption}\n")
        print("For more, visit: https://github.com/hitkalariya")

# Example function usage (for import)
# from generate_image_captions import load_image_captioning_model, generate_caption
# model, tokenizer, processor, device = load_image_captioning_model('path/to/model')
# caption = generate_caption('path/to/image.jpg', model, tokenizer, processor, device)