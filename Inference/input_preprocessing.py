import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle

class LeverageCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, loan_amount, annual_income):
        self.loan_amount = loan_amount
        self.annual_income = annual_income
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['leverage_ratio'] = X[self.loan_amount]/X[self.annual_income]
        X.insert(1, 'leverage_ratio', X.pop('leverage_ratio'))
        return X
    
class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, date_column,):
        self.date_column = date_column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_column] = X[self.date_column].apply(self.calculate_age)
        return X
        
    def calculate_age(self, date):
        if(pd.isna(date)):
            return 0
        
        date = pd.to_datetime(date, format='%m/%d/%Y %H:%M', exact=False)
        today = datetime.today()
        age = today.year - date.year - ((today.month, today.day) < (date.month, date.day))
        return age
    
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, purpose, description):
        self.purpose = purpose
        self.description = description
        
        
        self.percentage_mapping = {
            'Debt': 81.91841234840133,
            'Educational_Loan': 79.8076923076923,
            'Home_Loan': 86.60714285714286,
            'Personal_Loan': 88.05085555874199
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.purpose] = X[self.purpose].fillna('')
        X[self.description] = X[self.description].fillna('')
        X['text'] = X[self.purpose]+ X[self.description]
        X = X.drop(columns = [self.description])
        
        X[self.purpose] = X[self.purpose].map(self.percentage_mapping)
        return X
    
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, text_column):
        self.text_column = text_column
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.text_column] = X[self.text_column].apply(self._preprocess_text)
        return X
    
    def _preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        
        
        text = text.lower()
        # Tokenize into words
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
class TfidfConcatenator(BaseEstimator, TransformerMixin):
    def __init__(self, text_column):
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(max_features=30)

    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.text_column])
        return self

    def transform(self, X):
        X = X.copy()
        tfidf_matrix = self.vectorizer.transform(X[self.text_column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        X = X.drop(columns=[self.text_column])
        return pd.concat([X.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

