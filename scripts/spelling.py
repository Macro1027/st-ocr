from spellchecker import SpellChecker

def correct_spelling(text):
    spell = SpellChecker()
    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word:
            corrected_text.append(corrected_word)
            
    return ' '.join(corrected_text)
