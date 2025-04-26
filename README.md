# Artificial Illusion: The Art of Deepfake Technology

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
- **Presentation:** [Project Presentation]([presentation.pptx](https://github.com/BRAIN-Lab-AI/Artificial-Illusion-The-Art-of-Deepfake-Technology/blob/main/Deep_Learning_pt.pptx))
- **Report:** [Project Report]([report.pdf](https://github.com/BRAIN-Lab-AI/Artificial-Illusion-The-Art-of-Deepfake-Technology/blob/main/Artificial%20Illusion-The%20Art%20of%20Deepfake%20Technology_Final.pdf))
  

### Reference Paper
- Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). Mesonet: A compact facial video forgery detection network. 2018 IEEE International Workshop on Information Forensics and Security (WIFS), 1–7. https://doi.org/10.1109/wifs.2018.8630761

### Reference Dataset
- https://e.pcloud.link/publink/show?code=kZVFpdZ7nWSSuivFjjBxWoqvL1ilQssYhtX
- https://www.kaggle.com/datasets/xdxd003/ff-c23

## Project Technicalities
Deepfake Detection Technicalities
Terminologies
• Deepfake: Digitally manipulated media, typically videos, created using advanced artificial intelligence techniques.
• Binary Cross-Entropy Loss: A loss function used for binary classification tasks, measuring how well the predicted probabilities match the actual binary labels.
• AdamW Optimizer: An extension of the Adam optimizer that includes weight decay for improved generalization in training deep neural networks.
• Data Augmentation: Techniques to artificially expand training datasets by applying transformations such as rotations, cropping, and color adjustments to reduce overfitting.
• 3D Convolution: A convolution operation that captures temporal dynamics in videos by applying filters across both spatial and temporal dimensions.
• Validation Accuracy: A performance metric computed on a separate dataset to gauge model performance and help prevent overfitting.
• Epochs: The number of complete passes through the training dataset during model training.
• Batch Size: The number of samples processed simultaneously by the neural network before updating its weights.


### Problem Statements
As deepfake generator continue to spread on various fields the detection models are few and less compatible, though there do exist some strong models that have strengthen their root as best detectors for deepfake data such as Deepfake Detection Challenge (DFDC)[2]. To address this, we aim to improve the performance of the MesoNet model, which has shown promise in detecting deepfakes and Face2Face manipulations.

### Loopholes or Research Areas
• **Generalization Across Techniques:** Inconsistent performance across different deepfake generation techniques, indicating the need for more robust cross-technique generalization.
• **Temporal Feature Extraction:** Limited exploration of temporal feature extraction techniques, presenting opportunities for deeper investigation.
• **Computational Efficiency:** Significant computational resources are required for training and processing video sequences, highlighting the need for efficient model architectures.


### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign model architectures to improve computational efficiency and enhance temporal feature extraction.
2. **Advanced Loss Functions:** Integrate advanced loss functions (Binary cross entropy loss) to enhance model sensitivity to critical deepfake characteristics.
3. **Enhanced Data Augmentation:** Implement advanced video-specific data augmentation strategies to improve model robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced MesoNet Model by using TensorFlow to import Keras. The solution includes:

- **Modify the Archiricture:** Incorporates the video Sequnce to get the serial of video frams
- **Robust Loss Functions:** Use Binary Cross Entropy.
- **Enhance Optempization:** Update The Optimization to AdamW.
- **Initiate New Convolution:** Add new 3D convolotion model to handel the videos properly.

### Key Components
- **`DL_Project_Baseline_Model.ipynb`**: Jupiter Notebook file contains the baseline model components, in addition to enhanced model representation.
- **`DL_Project_Final_v3 (1).ipynb`**: Jupiter Notebook-based code contains the enhanced models in additon to training the models with benchmarking the performance on the new and old models. 
  
## Model Workflow
This workflow is designed to receive input as image/video fake/real and classify if this input is fake/ real.

1. **Input & Pre-processing:**
   - Accepts either a single image or a full video clip.
   - Videos are read frame-by-frame, lightly down-sampled and centre-cropped to a consistent aspect ratio.

2. **Face Detection & Alignment:**
   - Finds an initial face; a lightweight tracker keeps it centred across frames.
   - CNN face checks and landmark rotation give square, upright 256 × 256 patches.
   - All face coordinates are cached for repeatable experiments.
  
3. **Frame-level Classification:**
     - Three available CNN backbones: Meso4, MesoInception4 and Meso3D.
     - Both use AdamW, binary-cross-entropy and a scheduled weight-decay.
     - Each patch returns a probability p(real)..

4. **Video-level Aggregation & Decision:**
   - For videos, probabilities are averaged → (0, 1).
   - A threshold (default 0.5) converts that score into the final Real / Fake label.
5. **Training Monitoring:**
   - Custom callback prints epoch time, train/val loss & accuracy, current LR and WD.
   - Best weights (highest val_accuracy) are checkpointed to Google Drive/disk storage automatically.
6. Benchmark & Visual
  

## How to Run the Code

1. Upload the notebook to jupiter compatible compiler.
2. Install required libraries to run the code and model requirements.
3. Load saved weights.
4. run cell by cell in the notebook or run all cells at one time.
5. Note# because this code requires running on GPU, those a compatibility issue might arise if you are running from google colab or kaggle. Thus, install the pip libaraies then restart the session. This way you can run the rest of the code without compatibility issues. 
   


