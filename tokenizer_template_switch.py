import re
from transformers import AutoTokenizer

def extract_separators(template):
    """
    Extracts separators used in the tokenization template.
    """
    # Adjust the regex to correctly match the specific pattern between '{{' and '+ message["content"] +'
    pattern = r"\{\{\s*([^{}]+?)\s*\+ message\['content'\]"
    matches = re.findall(pattern, template)
    # Clean up any extra spaces and return the matches
    separators = [match.strip() for match in matches]

    if any("message['role']" in element for element in separators):
        roles = ["system", "user", "assistant"]
        separators_ = []
        for role in roles:
            separators_.append(separators[0].replace(" + message['role'] + ", role).replace("'",""))
        return separators_

    return separators

def detect_eos_token(jinja_template, tokenizer):
    if "<|im_end|>" in jinja_template:
        return "<|im_end|>"
    if "</s>" in jinja_template:
        return "</s>"
    if "eos_token" in jinja_template:
        return tokenizer.eos_token    
    else:
        return "<|endoftext|>"

def recover_messages(formatted_message, separators, eos_token):
    """
    Recovers the original messages from the formatted message string.
    """
    # Split the formatted message using the end-of-string token
    split_messages = formatted_message.split(eos_token)
    
    # Remove the last empty string if it exists due to a trailing separator
    if split_messages and split_messages[-1].strip() == '':
        split_messages.pop()

    # Prepare the list to hold the recovered messages
    recovered_messages = []

    # Define roles after the first message, alternating between "user" and "assistant"
    alternate_roles = ["user", "assistant"]
    
    # Iterate over the split messages
    for index, message_content in enumerate(split_messages):
        # Determine the role, starting with "system" for the first message
        # then alternating between "user" and "assistant" for subsequent messages
        if index == 0:
            role = "system"
        else:
            role = alternate_roles[(index - 1) % 2]

        # Clean the message content by removing leading/trailing whitespace and separators
        clean_content = message_content.strip()
        for separator in separators:
            clean_content = clean_content.replace(separator.strip("'"), '', 1).strip()

        # Append the cleaned message with its role to the list
        recovered_messages.append({"role": role, "content": clean_content})

    return recovered_messages

def recover_chat_messages(tokenized_chat, tokenizer):
    """
    Given a tokenized_chat string and a tokenizer, returns the list of message dictionaries.
    """
    jinja_template = tokenizer.chat_template
    separators = extract_separators(jinja_template)
    eos_token = eos_token = detect_eos_token(jinja_template, tokenizer)
    recovered_messages = recover_messages(tokenized_chat, separators, eos_token)
    return recovered_messages

# Example usage
if __name__ == "__main__":
    checkpoint = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False)
    print(tokenized_chat)
    
    recovered_messages = recover_chat_messages(tokenized_chat, tokenizer)
    print(recovered_messages)
