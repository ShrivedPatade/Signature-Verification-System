# Signature-Verification-System

## Overview
This repository contains the code and resources for a **Signature Verification System** using machine learning. The project leverages advanced algorithms and computer vision techniques to verify signatures in real-time from a live camera feed. This system is designed to enhance document security, prevent fraud, and streamline verification processes.

## Model link
https://drive.google.com/drive/folders/1VXrJWlcGxzcujAbOs1_hyhZtrVDGV2My?usp=drive_link
load the models into the 'SigVer' folder

## Key Features
- **Real-time Signature Verification**: Verifies signatures instantly from a camera feed.
- **Machine Learning Models**:
  - Convolutional Neural Networks (CNN)
  - Support Vector Machines (SVM)
  - Random Forest Classifier
  - XGBoost Classifier
- **Preprocessing Pipeline**: Includes signature extraction, noise reduction, and normalization for high accuracy.
- **Performance Metrics**: Evaluates models based on precision, recall, F1-score, and overall accuracy.

## Tech Stack
- **Programming Language**: Python
- **Frameworks & Libraries**:
  - TensorFlow/Keras
  - OpenCV
  - scikit-learn
  - XGBoost
  - Matplotlib/Seaborn for visualization
- **Dataset**: Custom dataset and publicly available signature datasets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CrazyBeastable/Signature-Verification-System.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Signature-Verification-System
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Train the model:
   ```bash
   python train_model.py
   ```
2. Test the system:
   ```bash
   python test_model.py
   ```
3. Run real-time verification:
   ```bash
   python realtime_verification.py
   ```

## Results
- **Accuracy Achieved**:
  - CNN: 98.5%
  - SVM: 96.2%
  - Random Forest: 94.8%
  - XGBoost: 97.3%

Visualization of results and model performance metrics can be found in the `results/` directory.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
Special thanks to the creators of the publicly available signature datasets and the contributors to the libraries used in this project.

---

## Contact
For any queries or feedback, feel free to reach out via [GitHub Issues](https://github.com/CrazyBeastable/Signature-Verification-System/issues).

---

# LinkedIn Post Summary
**Revolutionizing Signature Verification with Machine Learning**

Excited to share my latest project: a **Signature Verification System** that utilizes cutting-edge machine learning algorithms like CNN, SVM, Random Forest, and XGBoost. This system enables **real-time signature verification** through a live camera feed, providing a robust solution for fraud prevention and document security.

Key Highlights:
- Achieved 98.5% accuracy with CNN.
- Designed a comprehensive preprocessing pipeline for reliable results.
- Leveraged multiple classifiers for performance comparison.

This project is a step towards integrating AI into secure document processing workflows. Explore the GitHub repository for more details: [Signature Verification System](https://github.com/CrazyBeastable/Signature-Verification-System)

Feedback and collaboration opportunities are always welcome. Let’s innovate together!


