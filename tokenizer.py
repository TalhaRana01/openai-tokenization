import tiktoken



encoder = tiktoken.encoding_for_model("gpt-4o-mini")


# For checking  vab size in the encoder model 
print("Vacab size:", encoder.n_vocab)  # Vacab size: 200019

text = "The cat sat on the mat"

# Text to tokens
tokens = encoder.encode(text)

# print("Tokens:", tokens)  # Tokens: [976, 9059, 10139, 402, 290, 2450]


# Each word has tokens
# for token in tokens:
#     print(f"Token: {token}, Text: '{encoder.decode([token])}'")
    
    
# Token: 976, Text: 'The'
# Token: 9059, Text: ' cat'
# Token: 10139, Text: ' sat'
# Token: 402, Text: ' on'
# Token: 290, Text: ' the'
# Token: 2450, Text: ' mat'


# Token to text
decoded_text = encoder.decode(tokens)
print("Decoded Text:", decoded_text)  # Decoded Text: The cat sat on the


for txt in text.split():
    token = encoder.encode(txt)
    print(f"Text: '{txt}', Token: {token}")
