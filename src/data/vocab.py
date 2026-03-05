
# Vocab: 59 chars + <blank> = 60 total

original_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%")

original_letter_to_int = {c: i for i, c in enumerate(original_chars)}

# Add blank at 0
letter_to_int = {k: v + 1 for k, v in original_letter_to_int.items()}
letter_to_int["<blank>"] = 0

int_to_letter = {v: k for k, v in letter_to_int.items()}

vocab_size = len(letter_to_int)
blank_id = 0
