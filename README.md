## Sentiment Analysis of Bitcoin Data by Tweets Through Naive Bayes

### 📄 Abstract
This project explores the application of Naive Bayes in sentiment analysis of Bitcoin-related tweets. With Bitcoin's market sentiment influencing its volatility, this paper aims to classify tweets as positive, negative, or neutral using Natural Language Processing (NLP) and machine learning techniques. The proposed approach achieved an 87% classification accuracy, demonstrating its potential in aiding Bitcoin traders and analysts in decision-making.

### 🧑‍💻 Key Features
- Dataset Collection: Scraped and preprocessed a corpus of Bitcoin-related tweets.  
- Text Processing: Performed tokenization, stopword removal, and stemming for noise reduction.  
- Feature Extraction: Implemented TF-IDF vectorization for textual data representation.  
- Modeling: Used the Naive Bayes classifier to perform sentiment classification.  
- Evaluation: Measured model performance through metrics like accuracy, precision, recall, and F1-score.  
### 🛠️ Tools and Technologies
- Python  
- Natural Language Toolkit (NLTK)  
- scikit-learn  
- Pandas and NumPy  
- Matplotlib and Seaborn for visualization  
### 📊 Results
The model achieved:

- 87% Accuracy  
- Precision: 0.85  
- Recall: 0.88  
- F1-Score: 0.86  
The high accuracy and balanced metrics indicate the effectiveness of the Naive Bayes classifier in this application.


### 📂 Repository Structure
📁 Sentiment-Analysis-Bitcoin    
│
├── 📁 data    
│   ├── bitcoin_tweets.csv       # Processed dataset  
│   └── raw_tweets.csv          # Raw dataset before cleaning  
│
├── 📁 notebooks    
│   ├── EDA.ipynb                # Exploratory Data Analysis  
│   ├── Preprocessing.ipynb       # Text preprocessing and feature extraction  
│   └── Modeling.ipynb           # Naive Bayes implementation and evaluation  
│
├── 📁 results    
│   └── metrics.json             # Performance metrics  
│
├── 📜 LICENSE    
├── 📜 README.md                  # Project documentation  
└── 📜 requirements.txt          # Required Python packages  



### ⚙️ How to Run the Project
**1. Clone the repository:**  
- git clone https://github.com/your-username/Sentiment-Analysis-Bitcoin.git  
- cd Sentiment-Analysis-Bitcoin  
**2. Install dependencies:**  
- pip install -r requirements.txt  
**3. Run the notebooks in the notebooks folder step-by-step to explore data, preprocess it, and train the model.**  
**4. Visualize results in the results folder.**

### 📬 Connect
Feel free to reach out for any queries or collaborations:  
📧 ruchidattab@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/b-ruchi/)

