import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus: list[str]):
        self.bigram_counts = defaultdict(dict)
        self.build_model(corpus)

    def build_model(self, corpus: list[str]):
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                if next_word not in self.bigram_counts[current_word]:
                    self.bigram_counts[current_word][next_word] = 0
                self.bigram_counts[current_word][next_word] += 1

    def generate_text(self, start_word: str, length: int) -> str:
        current_word = start_word.lower()
        result = [current_word]
        for _ in range(length - 1):
            next_words_dict = self.bigram_counts.get(current_word)
            if not next_words_dict:
                break  
            words = list(next_words_dict.keys())
            weights = list(next_words_dict.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
            result.append(next_word)
            current_word = next_word
        return " ".join(result)
