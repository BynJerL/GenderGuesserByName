import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model_path = "name_gender_model.h5"
tokenizer_path = "tokenizer.pkl"

# Ensure the model and tokenizer files exist
if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    print("Error: Model or tokenizer file not found!")
    exit()

# Load the model
model = load_model(model_path)

# Load the tokenizer
with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

# Define the maximum length (this should match your training configuration)
max_length = 20  # Adjust to match the training max_length

# Define a function for predicting gender
def predict_gender(name):
    # Convert the name to a sequence
    sequence = tokenizer.texts_to_sequences([name])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict gender
    prediction = model.predict(padded_sequence)[0][0]
    return "Male" if prediction > 0.5 else "Female"

# CLI interface
if __name__ == "__main__":
    print("=== Name Gender Predictor ===")
    print("Type a name to predict gender or 'exit' to quit.")
    while True:
        # Get user input
        name = input("Enter a name: ").strip()
        if name.lower() == "exit":
            print("Goodbye!")
            break
        if not name:
            print("Please enter a valid name.")
            continue

        # Predict and display the result
        gender = predict_gender(name)
        print(f"The predicted gender for '{name}' is: {gender}")
