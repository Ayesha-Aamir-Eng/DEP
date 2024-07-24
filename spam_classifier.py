import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Ensure the required NLTK data is downloaded
nltk.download('stopwords')

# Define the paths to your directories
ham_dir = 'C:/Users/Bruce Wayne/Desktop/Project-Directory/Data/ham'
spam_dir = 'C:/Users/Bruce Wayne/Desktop/Project-Directory/Data/spam'

# Function to read files from a directory
def read_files_from_directory(directory):
    files_content = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    files_content.append(f.read())
            except PermissionError as e:
                print(f"Permission error accessing file: {file_path} - {e}")
            except Exception as e:
                print(f"Error reading file: {file_path} - {e}")
    return files_content

# Read files from both directories
ham_files = read_files_from_directory(ham_dir)
spam_files = read_files_from_directory(spam_dir)

# Combine the content and create labels
documents = ham_files + spam_files
labels = [0] * len(ham_files) + [1] * len(spam_files)

# Check the number of documents read
print(f"Number of ham documents: {len(ham_files)}")
print(f"Number of spam documents: {len(spam_files)}")
print(f"Total documents: {len(documents)}")

# Define the stop words
stop_words = list(stopwords.words('english'))

# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words=stop_words)

try:
    X = vectorizer.fit_transform(documents)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
except ValueError as e:
    print(f"An error occurred: {e}")

# Check if the vocabulary is empty
if len(vectorizer.vocabulary_) == 0:
    print("The vocabulary is empty. Check your preprocessing steps.")

# Continue with your analysis...
