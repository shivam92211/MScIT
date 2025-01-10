# # Use Colab

# from transformers import pipeline

# # Step 1: Load a Pre-trained Transformer Model
# chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium",  framework="pt", pad_token_id=50256)

# # Step 2: Start a Chat Session
# print("Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.")

# # Step 3: Loop for Chatting
# while True:
#     user_input = input("You: ")
    
#     if user_input.lower() == "exit":
#         print("Chatbot: Goodbye!")
#         break
    
#     # Generate a Response
#     response = chatbot(user_input, max_length=50, num_return_sequences=1)
#     print("Chatbot:", response[0]['generated_text'])






from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

while True:
  print("User:")
  new_user_input = input()
  if new_user_input.lower() == "exit":
    break

  # Process and encode user input
  new_user_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

  # Update chat history
  bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

  # Generate response
  chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

  # Decode and print response
  print("DialoGPT:", tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))