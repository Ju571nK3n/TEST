import pandas as pd
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from gensim.corpora import Dictionary
from gensim.models import LdaModel

def preprocess_data(df):
    # Filter only 'False-positive' events in 'Status' field
    fp_df = df[df["Policies"] != "Password Dissemination"].copy()

    # Combine 'Policies' with 'Violation Triggers'
    fp_df['combined'] = fp_df['Policies'] + ', ' + fp_df['Violation Triggers']

    # Define a preprocessing function for tokenization.
    def preprocess_combined(combined_str):
        return [token.strip() for token in combined_str.split(';')]

    # Tokenization
    processed_combined = fp_df['combined'].map(preprocess_combined)

    return processed_combined, fp_df["Status"]

def train_lda(processed_combined):
    # Create a dictionary (word-to-integer mapping)
    dictionary = Dictionary(processed_combined)

    # Convert to Bag-of-Words form
    corpus = [dictionary.doc2bow(token) for token in processed_combined]

    # Create and train LDA models
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary)
    # topic output
    for i, topic in lda.show_topics(formatted=False):
        print(f'Topic {i}:')
        for word, weight in topic:
            print(f'   {word}: {weight}')

    return lda, corpus, dictionary

# Use pandas' read_csv function to read a CSV file.
df = pd.read_csv('Incidents+(last+wrrk).csv', header=0, encoding='utf-8')

# data preprocessing
processed_combined, y = preprocess_data(df)
y = [0]*len(y)  # Add this line to set all y to 0

# LDA training
lda, corpus, dictionary = train_lda(processed_combined)

# Visualize LDA results
lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display, 'lda.html')

# list LDA results
for i, topic in lda.show_topics(formatted=False):
    print(f'Topic {i}:')
    for word, weight in topic:
        print(f'   {word}: {weight}')