import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function to transform text
def transform_text(text):
    '''Transforms the input text by:
       1. Converting it to lowercase, 
       2. Tokenzing, 
       3. Removing non-alphanumeric tokens,
       4. Removing stopwords & punctuation
       5. Stemming
    '''
    
    ps = PorterStemmer()
    text = text.lower() # converts to lowercase
    text = nltk.word_tokenize(text) # tokenizes the text
    text = [word for word in text if word.isalnum()] # removes non-alphanumeric tokens
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] # removes stopwords & punctuations
    text = [ps.stem(word) for word in text] # stems the words

    # Join the tokens back into a single string
    return " ".join(text)


# Function to preprocess the data
def preprocess_df(df, text_col = 'text', target_col = 'target'):
    '''Preprocess the df by:
       1. Encoding the target column, 
       2. Removes duplicates & 
       3. Transforms the text column
    '''
    try:
        logger.debug('Starting preprocessing of DataFrame')

        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logger.debug('Target column encoded')

        df = df.drop_duplicates(keep='first')
        logger.debug('Removed duplicate rows')

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text transofrmation: %s', e)
        raise


def main(text_col = 'text', target_col = 'target'):
    '''Main function to lod raw data, preprocess & save the processed data'''

    try:
        # Fetch data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed = preprocess_df(train_data, text_col, target_col)
        test_processed = preprocess_df(test_data, text_col, target_col)

        # Store the data in data/interim
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug('Processed data saved to: %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete data preprocessing: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()