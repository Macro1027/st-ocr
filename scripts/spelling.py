from spellchecker import Spellchecker

def correct_spelling(text):
    spell = Spellchecker()

    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word:
            corrected_text.append(corrected_word)      

    return ' '.join(corrected_text)

if __name__ == "__main__":
    print(correct_spelling("thiis is a testt"))
    