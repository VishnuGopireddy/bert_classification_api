'''
Perform text pre-processing
'''
import regex as re

def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], '')
    return text

def strip_smiles(text):
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in [':', ';']:
                words.append(word)
    return ' '.join(words)


def strip_all_entities(text):
    entity_prefixes = ['@','#']
#     for separator in string.punctuation:
#         if separator not in entity_prefixes :
#             text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def remove_numbers(text):
    words = []
    for word in text.split(' '):
        if not word.isalnum():
            words.append(word)
    return ' '.join(words)