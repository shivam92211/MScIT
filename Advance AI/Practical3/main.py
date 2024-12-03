# Use Colab

from transformers import pipeline

# Step 1: Load a Pre-trained Transformer Model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium",  framework="pt", pad_token_id=50256)

# Step 2: Start a Chat Session
print("Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.")

# Step 3: Loop for Chatting
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    # Generate a Response
    response = chatbot(user_input, max_length=50, num_return_sequences=1)
    print("Chatbot:", response[0]['generated_text'])
