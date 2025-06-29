import sys
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

# Load tokenizer data
with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# Build tokenizer
def all_captions(mapping):
    return [caption for captions in mapping.values() for caption in captions]

def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

tokenizer = create_tokenizer(all_captions(mapping))
max_length = 35

# Helper functions
def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    repeated_count = 0
    previous_word = None

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None or word == 'endseq' or (word == previous_word and repeated_count > 2):
            break

        in_text += ' ' + word

        if word == previous_word:
            repeated_count += 1
        else:
            repeated_count = 0
        previous_word = word

    return in_text.replace('startseq', '').strip()

# Load models
print("Loading models...")
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
caption_model = load_model("model.keras")

# Load and preprocess image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    features = vgg_model.predict(image, verbose=0)
    return features

# Main CLI logic
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_caption.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Processing image: {image_path}")
    try:
        features = extract_features(image_path)
        caption = predict_caption(caption_model, features, tokenizer, max_length)
        print(f"Generated Caption: {caption}")
    except Exception as e:
        print("Error:", str(e))
