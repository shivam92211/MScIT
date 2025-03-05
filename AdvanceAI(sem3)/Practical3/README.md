
# Chatbot Using Hugging Face Transformers

This project demonstrates how to build a simple chatbot using the **Hugging Face Transformers library**. The chatbot is based on the **DialoGPT-medium** model, a pre-trained conversational model developed by Microsoft.

---

## Features

- Utilizes the **DialoGPT-medium** model for conversational AI.
- Simple implementation using Hugging Face's `pipeline` for text generation.
- Interactive chat loop that processes user input and generates responses.
- Easily extensible to other pre-trained transformer models.

---

## Requirements

Before running the code, make sure you have the following installed:

- Python 3.7 or higher
- Hugging Face Transformers library

Install the necessary library using:
```bash
pip install transformers
```

---

## Usage

### 1. Load the Pre-trained Model

The chatbot uses the `DialoGPT-medium` model. You can replace it with any other model supported by Hugging Face's pipeline. For example:
```python
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium", framework="pt", pad_token_id=50256)
```

### 2. Start the Chatbot

The chatbot greets the user and engages in a conversation until the user types `exit`.

### Example:
```python
Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.
You: Hi there!
Chatbot: Hi! How can I assist you today?
You: Tell me a joke.
Chatbot: Why did the chicken cross the road? To get to the other side!
```

---

## Running the Chatbot in Google Colab

1. Open Google Colab: [Google Colab](https://colab.research.google.com/)
2. Copy and paste the code provided below into a new notebook cell:
    ```python
    from transformers import pipeline

    # Load the pre-trained model
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium", framework="pt", pad_token_id=50256)

    # Start chat session
    print("Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        print("Chatbot:", response[0]['generated_text'])
    ```
3. Run the notebook and start chatting with the bot.

---

## Notes

- You can experiment with different models by replacing `"microsoft/DialoGPT-medium"` in the code with any other compatible model name.
- The chatbot's responses are generated in a conversational context but may sometimes produce irrelevant or non-sensical responses due to limitations of the pre-trained model.

---

## To Do

- Add context handling for longer conversations.
- Experiment with other Hugging Face models for diverse responses.
- Implement a web or GUI interface for better user interaction.

---

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it.
