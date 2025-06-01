# Human-and-generative-voices-CNN-classification

A deep learning project to classify between human-recorded and AI-generated voices using Convolutional Neural Networks (CNN) on Mel-Spectrograms. This project demonstrates practical skills in audio preprocessing, data augmentation, and CNN model development using Python and TensorFlow.

🎯 Project Objective
The primary goal of this project is to develop a classification model that can distinguish between:

Human voice recordings

Machine-generated (AI-generated) voices from platforms such as Botnoi Voice

This project is especially relevant in the era of AI-generated deepfake audio, where distinguishing between real and fake voices is vital for security and content integrity.

🔍 Key Features
Custom audio dataset collection (both human and generated samples)

Data preprocessing: silence trimming, resampling, resizing

Feature extraction: Mel Spectrogram

Audio & Spectrogram augmentation (SpecAugment, Pitch/Time shift, Noise addition)

CNN-based model architecture for binary classification

Model evaluation using Accuracy, Precision, Recall, Confusion Matrix, ROC Curve

🧠 Technologies Used
Python 3.10

TensorFlow / Keras – deep learning model development

Librosa – audio processing

Matplotlib, Seaborn – data visualization

Pandas, NumPy – data handling

Scikit-learn – evaluation metrics

Jupyter Notebook and VS Code – development environment

⚙️ Model Architecture
Input: Mel Spectrogram images (resized to consistent dimensions)

Layers:

Conv2D + ReLU + MaxPooling

Dropout for regularization

Flatten + Dense (Fully Connected Layers)

Final Sigmoid layer for binary output

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metrics: Accuracy, Precision, Recall, ROC AUC

📈 Results
The final CNN model achieved:

Overall Accuracy: ~67% on the test set

Precision/Recall: High values for both human and machine voice classes

Robustness: Maintained performance across gender-specific human voices (male and female subsets)

See the output/ folder for confusion matrices, ROC curves, and model evaluation reports.

📚 Research Background
This project was developed as part of a Senior Project in Information Technology for Business at Chulalongkorn University, Thailand. It showcases a practical application of machine learning in the context of audio signal classification. For Further details, observe Paper_Report.pdf.

👤 Author
Pitchaya Teeraruttakul

🎓 Bachelor degree in Statistics and Information Technology for Business, Chulalongkorn University

📫 Linkin: https://www.linkedin.com/in/pitchaya-teerarutakul-1b285930b

📄 License
This project is open-source and available under the MIT License.
