import re

def clean_message(message):
    """
    Clean a message by converting it to lowercase, removing numbers, 
    and stripping punctuation and special characters.

    Parameters:
    ----------
    message : str
        The input message that needs to be cleaned.

    Returns:
    -------
    str
        The cleaned message, with all transformations applied.
    """
    # Convert to lowercase
    message = message.lower()
    # Remove numbers
    message = re.sub(r'\d+', '', message)
    # Remove punctuation and special characters (except for spaces)
    message = re.sub(r'[^\w\s]', '', message)
    return message

def count_word_in_message(word, message):
    """
    Count the occurrences of a specific word in a given message.

    Parameters:
    ----------
    word : str
        The word to search for in the message.
    message : str
        The message in which to count the occurrences of the word.

    Returns:
    -------
    int
        The count of occurrences of the specified word in the message.
      """
    len(re.findall(r'\b' + re.escape(word) + r'\b', message.lower()))
    return 