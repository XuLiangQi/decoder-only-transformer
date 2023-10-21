"""
Load characters from the given text file.
Input:
    dir: The directory to the text file.
Output:
    text: The characters from the text file.
"""
def load_text(dir):
    with open(dir, 'r', encoding='utf-8') as f:
        text = f.read()
    return text