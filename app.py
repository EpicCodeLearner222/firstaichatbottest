import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Define vocabulary
vocabulary = ["Hello", "there", "How", "are", "you", "doing", "today", "I", "am", "feeling", "great"]

# Function to convert sentence into one-hot vectors
def text_to_vector(text, vocab):
    words = text.split()
    vector = np.zeros(len(vocab))  # Create a zero vector of the same size as vocabulary
    
    for word in words:
        if word in vocab:
            vector[vocab.index(word)] = 1  # Set the index corresponding to the word in the vocabulary to 1
    
    return vector

# Training data
X_train = np.array([
    text_to_vector("Hello How are you", vocabulary),
    text_to_vector("How are you doing", vocabulary),
    text_to_vector("Hello there How are you doing", vocabulary),
    text_to_vector("What is your name", vocabulary),
    text_to_vector("I'm feeling great today", vocabulary),
    text_to_vector("Tell me about you", vocabulary),
    text_to_vector("What do you do", vocabulary),
])

y_train = [
    "Hi! I'm good, thanks!",  # "Hello How are you"
    "I'm feeling great, thanks!",  # "How are you doing"
    "Hey! How's it going?",  # "Hello there How are you doing"
    "I'm a chatbot!",  # "What is your name"
    "I'm happy today!",  # "I'm feeling great today"
    "I am a chatbot created to help!",  # "Tell me about you"
    "I assist with various tasks and can chat.",  # "What do you do"
]

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Test input and prediction
input_text = "Hello How are you"
input_vector = text_to_vector(input_text, vocabulary)

predicted_response = knn.predict([input_vector])
print("Predicted Response for '{}' : {}".format(input_text, predicted_response[0]))
