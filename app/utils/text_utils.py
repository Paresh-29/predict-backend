
# app/utils/text_utils.py

def clean_markdown(text: str) -> str:
    """
    Function to clean markdown formatting (removes asterisks and other markdown symbols).
    """
    return text.replace('*', '').replace('**', '')  # Removes asterisks for bold/italics
