# Image Captioning with Vision Transformer and BERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Advanced image captioning system using Vision Transformer (ViT) encoder and BERT decoder**

[![GitHub](https://img.shields.io/badge/GitHub-@hitkalariya-black?style=for-the-badge&logo=github)](https://github.com/hitkalariya)

</div>

---

## 🎯 Project Overview

This repository implements a state-of-the-art image captioning system that combines the power of Vision Transformers (ViT) for image understanding with BERT for natural language generation. The system is trained on the Flickr8k dataset and can generate human-like descriptions for any input image.

### Key Features

- **Vision Transformer Encoder**: Extracts rich visual features from images
- **BERT Decoder**: Generates natural language captions
- **End-to-End Training**: Complete pipeline from data preprocessing to model deployment
- **Easy Inference**: Simple script for generating captions on new images
- **Modular Design**: Clean, well-documented code structure

---

## 🏗️ Architecture

```
Input Image → ViT Encoder → Feature Extraction → BERT Decoder → Generated Caption
```

### Model Components

- **Encoder**: Vision Transformer (ViT-Base-16)
- **Decoder**: BERT-based text generation model
- **Framework**: Hugging Face Transformers
- **Training**: PyTorch with custom training loop

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hitkalariya/Image-Captioning-with-ViT-and-BERT.git
   cd Image-Captioning-with-ViT-and-BERT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**
   - Download [Flickr8k](https://www.kaggle.com/datasets/nunenuh/flickr8k) from Kaggle
   - Extract to your preferred location

---

## 🚀 Quick Start

### Training the Model

1. **Prepare your dataset structure**:
   ```
   /path/to/flickr8k/
   ├── Images/
   │   ├── 1000268201_693b08cb0e.jpg
   │   ├── 1000268201_693b08cb0e.jpg
   │   └── ...
   └── captions.txt
   ```

2. **Open the training notebook**:
   ```bash
   jupyter notebook train_image_captioning_model.ipynb
   ```

3. **Update dataset paths** in the notebook and run all cells

4. **Model will be saved** as `model.safetensors`

### Generating Captions

```python
from generate_image_captions import generate_caption

# Load your trained model
caption = generate_caption("path/to/your/image.jpg")
print(f"Generated caption: {caption}")
```

Or use the standalone script:

```bash
python generate_image_captions.py
```

---

## 📊 Performance & Results

### Sample Outputs

| Image | Generated Caption |
|-------|------------------|
| ![Sample 1](https://via.placeholder.com/200x150?text=Sample+Image) | "A white dog running in the park" |
| ![Sample 2](https://via.placeholder.com/200x150?text=Sample+Image) | "A smiling girl playing on the playground" |

### Evaluation Metrics

- **BLEU-4 Score**: 0.XX
- **METEOR Score**: 0.XX
- **CIDEr Score**: 0.XX

*Note: Actual metrics depend on training configuration and dataset size*

---

## 🛠️ Technical Details

### Model Configuration

- **Encoder**: ViT-Base-16 (86M parameters)
- **Decoder**: BERT-Base (110M parameters)
- **Image Size**: 224×224 pixels
- **Max Caption Length**: 128 tokens
- **Vocabulary Size**: 30,522 tokens

### Training Parameters

- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Epochs**: 10
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup with cosine decay

---

## 📁 Project Structure

```
Image-Captioning-with-ViT-and-BERT/
├── train_image_captioning_model.ipynb                       # Training notebook
├── generate_image_captions.py                               # Inference script
├── requirements.txt                                          # Dependencies
├── README.md                                                # This file
└── LICENSE                                                  # MIT License
```

---

## 🔧 Customization

### Using Different Datasets

The model can be adapted for other datasets like:
- MS COCO
- Conceptual Captions
- Custom datasets

### Hyperparameter Tuning

Key parameters to experiment with:
- Learning rate and scheduler
- Batch size and gradient accumulation
- Model architecture variants
- Data augmentation strategies

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Hit Kalariya**

- GitHub: [@hitkalariya](https://github.com/hitkalariya)
- Profile: https://github.com/hitkalariya
- Focus: AI/ML, Computer Vision, Deep Learning
- Expertise: Neural Networks, Large Language Models, Multimodal AI

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Google Research for Vision Transformer
- Flickr8k dataset creators
- Open source community

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/hitkalariya/Image-Captioning-with-ViT-and-BERT?style=social)](https://github.com/hitkalariya/Image-Captioning-with-ViT-and-BERT)

</div>
