from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load your fine-tuned GPT-2 model
MODEL_PATH = "fine_tuned_healthcare_gpt2"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

st.title("Healthcare Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a healthcare question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation history for the prompt
    conversation = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    conversation += "Assistant:"

    # Tokenize and generate response from the fine-tuned GPT-2 model
    inputs = tokenizer(conversation, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
