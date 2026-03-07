import contractions
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer             # NLP Preprocessing

#Lemmatizer will reduce a word to its base form. For example words like playing, played, plays will have a base word 'play'
lemmatizer = WordNetLemmatizer()

# Keep negations
stop_words = ENGLISH_STOP_WORDS - {"no", "not", "nor"}

def text_cleaning(text):
    #text =  pd.Series(text)

    # 1. Expanding the contracted terms
    text = contractions.fix(text)
    
    # 2. Replace URLs, Emails, User mentions, Hashtags
    text = re.sub(r'http\S+|www\S+', 'URL', text)         # URLs
    text = re.sub(r'\S+@\S+', 'EMAIL', text)              # Emails
    text = re.sub(r'@\w+', 'USER', text)                  # User mentions
    text = re.sub(r'#(\w+)', r'\1', text)                 # Hashtags (keep word)

    # 3. remove punctuation except !?'#@* 
    keep_punct = "!?'#@*"
    punct_to_remove = ''.join([p for p in string.punctuation if p not in keep_punct])
    text = re.sub(f"[{re.escape(punct_to_remove)}]", " ", text)

    # 4. remove newline characters
    text = text.replace("\n", " ")

    # 5. Remove HTML artifact
    text = re.sub(r'\bnbsp\b', ' ', text)

    # 6. Instead of removing digits from text file we will replace it with 'NUM' to capture toxicity
    #Ex - sh1t or a55hole will become shNUMt or aNUMhole
    text = re.sub(r'\d+', 'NUM', text)

    # 7. splitting sentence into tokens
    text_tokens = text.split()

    # 8. converting the word to its meaningful base form
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text_tokens]

    # 9. removing stopword from sentence
    cleaned_tokens = [w for w in lemmatized_text if w not in stop_words]

    # 10. Rejoin and clean extra spaces
    text = ' '.join(cleaned_tokens)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    