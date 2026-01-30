from spellchecker import SpellChecker

spell = SpellChecker()

# Take input from user
text = input("Enter a sentence: ")

# Split sentence into words
words = text.split()

corrected_words = []

for word in words:
    if word in spell:
        corrected_words.append(word)
    else:
        corrected_words.append(spell.correction(word))

# Join corrected words
corrected_text = " ".join(corrected_words)

print("Corrected Text:", corrected_text)
