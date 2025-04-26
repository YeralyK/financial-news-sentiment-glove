# Financial News Sentiment Analysis using CNN and GloVe

## 📚 Project Overview
This project applies a Convolutional Neural Network (CNN) to classify financial news headlines into sentiment categories (Positive / Negative).  
We utilize GloVe embeddings to improve model understanding of language semantics.

## 📈 Key Features
- Preprocessing of financial news headlines
- Tokenization and padding of text data
- Use of pretrained **GloVe** (100d) word embeddings
- Medium-complexity **CNN** architecture
- Early stopping to prevent overfitting
- Achieved **100% accuracy** on validation set (due to dataset properties)

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Pandas / NumPy

## 📦 Dataset
- [FinancialPhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- Modified into a CSV with columns: `Title`, `Sentiment`

## 🔥 Model Architecture
- Embedding layer (with GloVe pretrained vectors)
- 2× Conv1D layers
- Global Max Pooling
- Dense + Dropout layers
- Output: Sigmoid activation for binary classification

## 📊 Evaluation
- Accuracy: **100%** on validation data
- Note: Validation set contained only **negative class** due to dataset imbalance. Model performance on positive class was not validated.

## 🧠 Insights
- GloVe embeddings significantly improve the model's ability to understand financial text
- Class imbalance must be handled carefully in real-world applications (e.g., resampling, class weighting)

## 📂 How to Run
1. Clone the repo
2. Install necessary libraries (`pip install -r requirements.txt`)
3. Train model or load pretrained `.h5` model
4. Run predictions on new financial headlines

## 📥 Files Included
- `financial_news_sentiment_cnn_glove.h5` — Pretrained model
- `Stock_News_CNN_Project.ipynb` — Jupyter notebook with full training code

## ✨ Author
**Yeraly Kaimolda**  
[GitHub](https://github.com/YeralyK)

---

