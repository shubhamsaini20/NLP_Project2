#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd


# In[85]:


df = pd.read_csv("C:\\Users\\SHUBHAM SAINI\\Desktop\\trustpilot_scraper-main\\en_m.csv")


# In[86]:


df.head(5)


# In[87]:


df['ratingValue'].info


# In[88]:


import pandas as pd

# Assuming your DataFrame is named 'df'
# Add a new column 'cleaned_company' to store the cleaned company names

def company_name(url):
    # Extract the domain from the URL
    domain = url.split('.')[1]

    # Replace specific domain names as per your rules
    if domain == 'onecallinsurance':
        return 'One Call Insurance'
    elif domain == 'rac':
        return 'RAC'
    else:
        # For other domains, you might want to implement additional rules or leave them as is
        return domain.capitalize()  # Capitalize the first letter

# Apply the function to the 'company' column
df['company_name'] = df['company'].apply(company_name)

# Print the updated DataFrame
print(df[['company', 'company_name']])


# In[89]:


df.head(4)


# ### Dropping unneccessary columns
# #### We dropped two columns named 'url' & 'company'

# In[90]:


df.drop('company', axis=1, inplace=True)


# In[ ]:





# In[91]:


df.drop('url', axis=1, inplace=True)


# In[92]:


df.head(2)


# In[93]:


# removing duplicates 
df = df.drop_duplicates()


# In[94]:


#converting date column to datetime column
df['date'] = pd.to_datetime(df['date'])


# In[95]:


df.head(5)


# In[96]:


df.info()     


# In[97]:


df.describe()


# # Visualization

# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot a histogram of ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['ratingValue'], bins=5, kde=False, color='skyblue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Plot a bar chart of average ratings by company
average_ratings_by_company = df.groupby('company_name')['ratingValue'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
average_ratings_by_company.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Ratings by Company')
plt.xlabel('Company')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[100]:


plt.figure(figsize=(12, 8))
sns.violinplot(x='company_name', y='ratingValue', data=df, palette='viridis')
plt.title('Rating Distributions by Company')
plt.xlabel('Company')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[107]:


import plotly.express as px

fig = px.scatter(df, x='date', y='ratingValue', color='company_name', hover_data=['headline', 'name'],
                 title='Interactive Scatter Plot with Tooltip')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rating')
fig.show()



# # Supervised Learning ML tech

# In[130]:


# Text Preprocessing:
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Text preprocessing function
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the processed words
    processed_text = ' '.join(words)

    return processed_text

# Apply text preprocessing to the 'review' column
df['processed_review'] = df['review'].apply(preprocess_text)

# Print the first few rows of the processed data
print(df[['processed_review', 'ratingValue']].head())


# In[129]:


#Model Training 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['ratingValue'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a simple classifier (e.g., Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))


# # Streamlit Application

# In[133]:


import streamlit as st

# Streamlit application
st.title("Sentiment Analysis and Detailed Information Extraction")

# User input for sentiment analysis
user_input = st.text_area("Enter your review here:")

if user_input:
    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Convert to TF-IDF features
    user_tfidf = vectorizer.transform([processed_input])

    # Predict sentiment
    sentiment_prediction = classifier.predict(user_tfidf)[0]

    # Display sentiment prediction
    st.write(f"Sentiment Prediction: {sentiment_prediction}")

    # Display detailed information (dummy information, replace with actual logic)
    if sentiment_prediction > 3:
        st.write("Detailed Information:")
        st.write("Positive review details - e.g., information about food, service, etc.")
    else:
        st.write("Negative review details - e.g., reasons for dissatisfaction")

# Additional functionalities (not implemented in this example)
# - Restaurant summary based on restaurant name
# - QA system for restaurant recommendations

# Run the Streamlit app
# Command to run: streamlit run your_app_name.py


# In[132]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




