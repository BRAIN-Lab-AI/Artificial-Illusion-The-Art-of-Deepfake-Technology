# Artificial Illusion: The Art of Deepfake Technology

Below is a template for another sample project. Please follow this template.
# [Deep Learning Project Template] Enhanced Stable Diffusion: A Deep Learning Approach for Artistic Image Generation

## Introduction
Artificial Intelligence (AI) has become one of the most widely discussed topics across various fields in recent years, sparking both curiosity and concern due to its rapid advancements. Among AI-driven technologies, deepfake stands out as a powerful yet controversial tool. It enables the creation of synthetic media that can convincingly impersonate real individuals, objects, or events, often with striking realism.
Due to its ability to generate highly realistic fake content with minimal effort, deepfake technology has significantly impacted the media industry. Unfortunately, its accessibility also makes it a tool for malicious purposes, such as spreading misinformation, manipulating public opinion, and committing fraud. Scammers, for instance, can easily create faked media to impersonate real individuals, deceiving the public and spreading false narratives.
Considering these concerns, we’ve searched for a way to detect these falsifications and found a suitable proposed methodology in the research paper “MesoNet: A Compact Facial Video Forgery Detection Network” [1].

## Project Metadata
### Authors
- **Team:** Jawaher Mohammed Alkhamis (g202403980)
            Dhoha Ahmed Almubayedh (g202403920)
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [1] ] Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). Mesonet: A compact facial video forgery detection network. 2018 IEEE International Workshop on Information Forensics and Security (WIFS), 1–7. https://doi.org/10.1109/wifs.2018.8630761

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities
Deepfake Detection Technicalities
Terminologies
•	Deepfake: Digitally manipulated media, typically videos, created using advanced artificial intelligence techniques.
•	Binary Cross-Entropy Loss: A loss function used for binary classification tasks, measuring how well the predicted probabilities match the actual binary labels.
•	AdamW Optimizer: An extension of the Adam optimizer that includes weight decay for improved generalization in training deep neural networks.
•	Data Augmentation: Techniques to artificially expand training datasets by applying transformations such as rotations, cropping, and color adjustments to reduce overfitting.
•	3D Convolution: A convolution operation that captures temporal dynamics in videos by applying filters across both spatial and temporal dimensions.
•	Validation Accuracy: A performance metric computed on a separate dataset to gauge model performance and help prevent overfitting.
•	Epochs: The number of complete passes through the training dataset during model training.
•	Batch Size: The number of samples processed simultaneously by the neural network before updating its weights.


### Problem Statements
As deepfake generator continue to spread on various fields the detection models are few and less compatible, though there do exist some strong models that have strengthen their root as best detectors for deepfake data such as Deepfake Detection Challenge (DFDC)[2]. To address this, we aim to improve the performance of the MesoNet model, which has shown promise in detecting deepfakes and Face2Face manipulations.

### Loopholes or Research Areas
•	**Generalization Across Techniques:** Inconsistent performance across different deepfake generation techniques, indicating the need for more robust cross-technique generalization.
•	**Temporal Feature Extraction:** Limited exploration of temporal feature extraction techniques, presenting opportunities for deeper investigation.
•	**Computational Efficiency:** Significant computational resources are required for training and processing video sequences, highlighting the need for efficient model architectures.


### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign model architectures to improve computational efficiency and enhance temporal feature extraction.
2. **Advanced Loss Functions:** Integrate advanced loss functions (Binary cross entropy loss) to enhance model sensitivity to critical deepfake characteristics.
3. **Enhanced Data Augmentation:** Implement advanced video-specific data augmentation strategies to improve model robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.


