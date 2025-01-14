import random
import string
import nltk
from nltk.corpus import words, brown
import math
import time
from collections import Counter
#import numpy as np
from itertools import combinations
import sys

class DecipherText(object): # Do not change this
    def decipher(self, ciphertext): # Do not change this
        """Decipher the given ciphertext"""

        # Define the ciphertext alphabet and the plaintext alphabet

        CIPHERTEXT_ALPHABET = ['1','2','3','4','5','6','7','8','9','0','@','#','$', 'z','y','x','w','v','u','t','s','r','q','p','o','n']
        PLAINTEXT_ALPHABET = list(string.ascii_uppercase)  # ['A','B','C',...'Z']

        # Define English frequency orders (unigram, bigram, trigram)

        ENGLISH_UNIGRAM_FREQ_ORDER = "ETAONRISHDLFCMUGYPWBVKJXQZ"

        # 3. Load NLTK corpora: words, brown (for bigrams and trigrams)

        def load_nltk_corpora():
            """
            Ensure that necessary NLTK corpora are downloaded.
            """
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                print("NLTK 'words' corpus not found. Downloading now...")
                nltk.download('words')

            try:
                nltk.data.find('corpora/brown')
            except LookupError:
                print("NLTK 'brown' corpus not found. Downloading now...")
                nltk.download('brown')

        # --------------------------------------------------------------------
        # 4. Load bigram frequencies from Brown corpus
        # --------------------------------------------------------------------
        def load_bigram_freq():
            """
            Load bigram frequencies from NLTK's Brown corpus.
            Returns a dictionary with bigram tuples as keys and log probabilities as values.
            """
            corpus = brown.words()
            corpus = [word.upper() for word in corpus if word.isalpha()]

            bigram_counts = Counter()
            total_bigrams = 0
            for word in corpus:
                if len(word) < 2:
                    continue
                bigram_counts.update(zip(word, word[1:]))
                total_bigrams += len(word) - 1

            # Convert counts to log probabilities
            bigram_freq = {bg: math.log(count / total_bigrams) for bg, count in bigram_counts.items()}

            # Assign a small probability to unseen bigrams
            unseen_prob = math.log(0.01 / total_bigrams)
            return bigram_freq, unseen_prob

        # --------------------------------------------------------------------
        # 5. Load trigram frequencies from Brown corpus
        # --------------------------------------------------------------------
        def load_trigram_freq():
            """
            Load trigram frequencies from NLTK's Brown corpus.
            Returns a dictionary with trigram tuples as keys and log probabilities as values.
            """
            corpus = brown.words()
            corpus = [word.upper() for word in corpus if word.isalpha()]

            trigram_counts = Counter()
            total_trigrams = 0
            for word in corpus:
                if len(word) < 3:
                    continue
                trigram_counts.update(zip(word, word[1:], word[2:]))
                total_trigrams += len(word) - 2

            # Convert counts to log probabilities
            trigram_freq = {tg: math.log(count / total_trigrams) for tg, count in trigram_counts.items()}

            # Assign a small probability to unseen trigrams
            unseen_prob = math.log(0.001 / total_trigrams)
            return trigram_freq, unseen_prob

        # --------------------------------------------------------------------
        # 6. Load the English dictionary using NLTK
        # --------------------------------------------------------------------
        def load_dictionary():
            """
            Load the English dictionary from NLTK's words corpus.
            Returns a set of lowercase words for quick lookup.
            """
            word_list = words.words()
            # Filter words to include only alphabetic characters and convert to lowercase
            english_dict = set(word.lower() for word in word_list if word.isalpha())
            return english_dict

        # --------------------------------------------------------------------
        # 7. Utility: Seeding the Random Number Generator for Reproducibility
        # --------------------------------------------------------------------
        def seed_random_generators(seed=42):
            """
            Seed the random number generators for reproducibility.
            """
            random.seed(seed)
            #np.random.seed(seed)

        # --------------------------------------------------------------------
        # 8. Utility: Scoring function
        #    - Incorporates unigram, bigram, and trigram frequencies
        #    - Counts dictionary words
        #    - Adjustable weights
        # --------------------------------------------------------------------
        class Scorer:
            def __init__(self, dictionary, bigram_freq, trigram_freq, unseen_prob_bigram, unseen_prob_trigram, weights=(20, 10, 5)):
                self.dictionary = dictionary
                self.bigram_freq = bigram_freq
                self.trigram_freq = trigram_freq
                self.unseen_prob_bigram = unseen_prob_bigram
                self.unseen_prob_trigram = unseen_prob_trigram
                self.weights = weights  # (dict_weight, trigram_weight, bigram_weight)
                self.cache = {}

            def score(self, decoded_text):
                """
                Compute a combined score based on:
                    - Number of valid dictionary words
                    - Bigram frequencies
                    - Trigram frequencies
                Higher scores indicate better mappings.
                """
                if decoded_text in self.cache:
                    return self.cache[decoded_text]

                # Dictionary word score
                words_in_text = ''.join([char if char.isalpha() else ' ' for char in decoded_text]).lower().split()
                dict_score = sum(1 for word in words_in_text if word in self.dictionary)

                # Bigram score
                # Bigram score
                decoded_text_upper = decoded_text.upper()
                bigrams = zip(decoded_text_upper, decoded_text_upper[1:])
                bigram_log_probs = [
                    self.bigram_freq.get(bg, self.unseen_prob_bigram) 
                    for bg in bigrams 
                    if bg[0] in string.ascii_uppercase and bg[1] in string.ascii_uppercase
                ]
                bigram_score = sum(bigram_log_probs) if bigram_log_probs else self.unseen_prob_bigram

                # Trigram score
                trigrams = zip(decoded_text_upper, decoded_text_upper[1:], decoded_text_upper[2:])
                trigram_log_probs = [
                    self.trigram_freq.get(tg, self.unseen_prob_trigram) 
                    for tg in trigrams 
                    if tg[0] in string.ascii_uppercase and tg[1] in string.ascii_uppercase and tg[2] in string.ascii_uppercase
                ]
                trigram_score = sum(trigram_log_probs) if trigram_log_probs else self.unseen_prob_trigram

                # Combine scores with adjustable weights
                dict_weight, trigram_weight, bigram_weight = self.weights
                total_score = dict_score * dict_weight + trigram_score * trigram_weight + bigram_score * bigram_weight

                # Cache the result
                self.cache[decoded_text] = total_score

                return total_score

        # --------------------------------------------------------------------
        # 9. Frequency analysis: Build an initial mapping
        #    - Enhanced using bigrams and trigrams
        # --------------------------------------------------------------------
        def build_initial_mapping(ciphertext):
            """
            1. Count frequency of each symbol in ciphertext.
            2. Sort symbols by descending frequency.
            3. Map them to letters in ENGLISH_UNIGRAM_FREQ_ORDER.
            4. Return that initial mapping as a dict: {cipher_symbol -> plain_letter}
            """
            # Count frequencies using Counter for efficiency
            freq_dict = Counter([ch for ch in ciphertext if ch in CIPHERTEXT_ALPHABET])

            # Sort by descending frequency
            sorted_symbols = [item[0] for item in freq_dict.most_common()]

            # Build mapping
            initial_map = {}
            for i, symbol in enumerate(sorted_symbols):
                if i < len(ENGLISH_UNIGRAM_FREQ_ORDER):
                    initial_map[symbol] = ENGLISH_UNIGRAM_FREQ_ORDER[i]
                else:
                    # If more symbols, map to remaining letters or '_'
                    initial_map[symbol] = '_'

            # For any ciphertext symbol not encountered in freq_dict, map to something
            assigned_letters = set(initial_map.values())
            leftover_letters = [l for l in PLAINTEXT_ALPHABET if l not in assigned_letters]
            leftover_index = 0
            for sym in CIPHERTEXT_ALPHABET:
                if sym not in initial_map:
                    if leftover_index < len(leftover_letters):
                        initial_map[sym] = leftover_letters[leftover_index]
                        leftover_index += 1
                    else:
                        initial_map[sym] = '_'

            return initial_map

        # --------------------------------------------------------------------
        # 10. Decode function: given a mapping, decode the ciphertext
        # --------------------------------------------------------------------
        def decode(ciphertext, mapping):
            """
            Convert each symbol in ciphertext to its mapped letter.
            Preserve case for letters, keep other punctuation/spaces as-is.
            """
            # Using list comprehension for speed
            decoded = [
                mapping[ch] if ch in mapping and mapping[ch] != '_' else ch
                for ch in ciphertext
            ]
            return ''.join(decoded)

        # --------------------------------------------------------------------
        # 11. Hill-Climbing approach to refine the mapping
        # --------------------------------------------------------------------
        def hill_climb(ciphertext, scorer, initial_map, max_iterations=10000, early_stop_threshold=1000):
            """
            Attempt to improve the mapping with a hill-climbing strategy:

            1. Start from initial_map.
            2. Repeatedly try swapping two ciphertext symbols' mappings.
            3. If the new mapping improves the score, keep it.
            4. Return the best mapping found along with its score.

            Enhancements:
                - Early stopping if no improvement after certain iterations.
            """
            best_map = initial_map.copy()
            best_decoded = decode(ciphertext, best_map)
            best_score = scorer.score(best_decoded)
            no_improve_counter = 0

            for iteration in range(max_iterations):
                # Randomly pick two ciphertext symbols to swap their mapped letters
                c1, c2 = random.sample(CIPHERTEXT_ALPHABET, 2)

                # Swap the mappings in a copy
                new_map = best_map.copy()
                new_map[c1], new_map[c2] = new_map[c2], new_map[c1]

                # Decode and score
                candidate_decoded = decode(ciphertext, new_map)
                candidate_score = scorer.score(candidate_decoded)

                # If the new score is better, accept the swap
                if candidate_score > best_score:
                    best_map = new_map
                    best_score = candidate_score
                    best_decoded = candidate_decoded
                    no_improve_counter = 0
                    # Optionally, print progress
                    if (iteration + 1) % 1000 == 0:
                        #print(f"Hill-Climb Iteration {iteration + 1}: Improved score to {best_score}")
                        pass
                else:
                    no_improve_counter += 1
                    if no_improve_counter >= early_stop_threshold:
                        #print(f"Hill-Climb: No improvement in the last {early_stop_threshold} iterations. Stopping early.")
                        break

            return best_map, best_score

        # --------------------------------------------------------------------
        # 12. Simulated Annealing approach to refine the mapping
        # --------------------------------------------------------------------
        def simulated_annealing(ciphertext, scorer, initial_map, max_iterations=10000, initial_temp=100.0, cooling_rate=0.001, min_temperature=1e-4):
            """
            Attempt to improve the mapping with a simulated annealing strategy:

            1. Start from initial_map.
            2. At each step, swap two ciphertext symbols' mappings.
            3. Accept the swap based on the change in score and the current temperature.
            4. Gradually cool down the temperature.
            5. Return the best mapping found along with its score.

            Enhancements:
                - Early stopping based on temperature.
            """
            current_map = initial_map.copy()
            current_decoded = decode(ciphertext, current_map)
            current_score = scorer.score(current_decoded)

            best_map = current_map.copy()
            best_score = current_score
            best_decoded = current_decoded

            temperature = initial_temp
            alpha = cooling_rate  # cooling rate

            for iteration in range(max_iterations):
                if temperature < min_temperature:
                    #print(f"Simulated Annealing: Temperature below {min_temperature}. Stopping early.")
                    break

                # Randomly pick two ciphertext symbols to swap their mapped letters
                c1, c2 = random.sample(CIPHERTEXT_ALPHABET, 2)

                # Swap the mappings in a copy
                new_map = current_map.copy()
                new_map[c1], new_map[c2] = new_map[c2], new_map[c1]

                # Decode and score
                candidate_decoded = decode(ciphertext, new_map)
                candidate_score = scorer.score(candidate_decoded)

                # Calculate score difference
                score_diff = candidate_score - current_score

                # Decide whether to accept the swap
                if score_diff > 0 or random.random() < math.exp(score_diff / temperature):
                    current_map = new_map
                    current_score = candidate_score
                    current_decoded = candidate_decoded

                    # Update best found so far
                    if current_score > best_score:
                        best_map = current_map.copy()
                        best_score = current_score
                        best_decoded = current_decoded
                        # Optionally, print progress
                        if (iteration + 1) % 1000 == 0:
                            #print(f"SA Iteration {iteration + 1}: Improved score to {best_score}")
                            pass

                # Cool down the temperature
                temperature *= (1 - alpha)

            return best_map, best_score

        # --------------------------------------------------------------------
        # 13. Genetic Algorithms approach to refine the mapping
        # --------------------------------------------------------------------
        def genetic_algorithm(ciphertext, scorer, population_size=100, generations=200, mutation_rate=0.1, elitism=True, elitism_count=5):
            """
            Implement a Genetic Algorithm to optimize the mapping.

            Steps:
                1. Initialize a population of random mappings.
                2. Evaluate the fitness of each mapping.
                3. Select the top-performing mappings as parents.
                4. Perform crossover and mutation to create offspring.
                5. Repeat for a number of generations.
                6. Return the best mapping found.

            Enhancements:
                - Elitism to retain top mappings across generations.
                - Increased number of generations for better convergence.
            """
            # Initialize population with random mappings
            population = [random_mapping() for _ in range(population_size)]

            for generation in range(generations):
                # Evaluate fitness
                fitness_scores = [scorer.score(decode(ciphertext, mapping)) for mapping in population]

                # Select top performers (e.g., top 50%)
                sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
                top_indices = sorted_indices[:population_size//2]
                parents = [population[i] for i in top_indices]

                # Generate offspring through crossover
                offspring = []
                while len(offspring) < population_size//2:
                    parent1, parent2 = random.sample(parents, 2)
                    child = crossover(parent1, parent2)
                    offspring.append(child)

                # Apply mutation
                for child in offspring:
                    if random.random() < mutation_rate:
                        mutate(child)

                # Apply elitism
                if elitism:
                    #elites = [population[i] for i in np.argsort(fitness_scores)[-elitism_count:]]
                    #population = elites + offspring
                    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                    elites = [population[i] for i in sorted_indices[-elitism_count:]]
                    population = elites + offspring
                else:
                    population = parents + offspring

                # Optionally, print progress
                best_score = max(fitness_scores)
                #print(f"Generation {generation + 1}: Best Score = {best_score}")

            # Final evaluation
            final_scores = [scorer.score(decode(ciphertext, mapping)) for mapping in population]
            best_score = max(final_scores)
            best_index = final_scores.index(best_score)
            best_map = population[best_index]

            return best_map, best_score

        def random_mapping():
            """
            Generate a random mapping from ciphertext symbols to plaintext letters.
            Ensures each plaintext letter is assigned to only one ciphertext symbol.
            """
            letters = PLAINTEXT_ALPHABET.copy()
            random.shuffle(letters)
            mapping = {}
            for i, symbol in enumerate(CIPHERTEXT_ALPHABET):
                if i < len(letters):
                    mapping[symbol] = letters[i]
                else:
                    mapping[symbol] = '_'  # Assign '_' to any extra symbols, if any
            return mapping

        def crossover(parent1, parent2):
            """
            Perform crossover between two parent mappings to produce a child mapping.
            Uses single-point crossover.
            """
            child = {}
            crossover_point = random.randint(1, len(CIPHERTEXT_ALPHABET)-1)
            for i, symbol in enumerate(CIPHERTEXT_ALPHABET):
                if i < crossover_point:
                    child[symbol] = parent1[symbol]
                else:
                    child[symbol] = parent2[symbol]
            # Resolve duplicates to maintain one-to-one mapping
            child = resolve_duplicates(child)
            return child

        def mutate(mapping):
            """
            Perform mutation on a mapping by swapping two ciphertext symbols' mappings.
            """
            c1, c2 = random.sample(CIPHERTEXT_ALPHABET, 2)
            mapping[c1], mapping[c2] = mapping[c2], mapping[c1]

        def resolve_duplicates(child):
            """
            Ensure that the child mapping maintains a one-to-one mapping.
            Resolve duplicates by assigning unused letters to conflicting symbols.
            """
            used_letters = set()
            duplicates = {}
            for symbol, letter in child.items():
                if letter in used_letters:
                    duplicates[symbol] = letter
                else:
                    used_letters.add(letter)

            # Find unused letters
            unused_letters = [l for l in PLAINTEXT_ALPHABET if l not in used_letters]

            for symbol in duplicates:
                if unused_letters:
                    child[symbol] = unused_letters.pop()
                else:
                    child[symbol] = '_'  # Assign '_' if no letters are left

            return child

        # --------------------------------------------------------------------
        # 14. Beam Search Integration
        # --------------------------------------------------------------------
        def beam_search(ciphertext, scorer, initial_map, beam_width=10, max_steps=200):
            """
            Perform Beam Search to optimize the mapping.

            Parameters:
                ciphertext (str): The ciphertext to decrypt.
                scorer (Scorer): The scoring object.
                initial_map (dict): The initial mapping.
                beam_width (int): Number of top candidates to retain at each step.
                max_steps (int): Maximum number of search steps.

            Returns:
                best_map (dict): The best mapping found.
                best_score (float): The score of the best mapping.
            """
            beam = [(initial_map, scorer.score(decode(ciphertext, initial_map)))]

            for step in range(max_steps):
                all_candidates = []
                for mapping, score in beam:
                    # Generate all possible single swaps
                    for c1, c2 in combinations(CIPHERTEXT_ALPHABET, 2):
                        new_map = mapping.copy()
                        new_map[c1], new_map[c2] = new_map[c2], new_map[c1]
                        decoded = decode(ciphertext, new_map)
                        new_score = scorer.score(decoded)
                        all_candidates.append((new_map, new_score))

                # Select top beam_width candidates
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                new_beam = all_candidates[:beam_width]

                # Check for improvement
                if new_beam[0][1] <= beam[0][1]:
                    #print(f"Beam Search: No improvement at step {step}. Stopping early.")
                    break

                beam = new_beam
                #print(f"Beam Search Step {step + 1}: Best Score = {beam[0][1]}")

            best_map, best_score = beam[0]
            return best_map, best_score

        # --------------------------------------------------------------------
        # 15. Multiple Restarts Implementation
        # --------------------------------------------------------------------
        def multiple_restarts(ciphertext, scorer, initial_map, restarts=5, hill_iterations=10000, sa_iterations=10000, ga_generations=200, beam_width=10, beam_steps=200):
            """
            Perform multiple restarts of the optimization to avoid local maxima.

            Parameters:
                ciphertext (str): The ciphertext to decrypt.
                scorer (Scorer): The scoring object.
                initial_map (dict): The initial mapping.
                restarts (int): Number of optimization restarts.
                hill_iterations (int): Number of iterations for Hill-Climbing.
                sa_iterations (int): Number of iterations for Simulated Annealing.
                ga_generations (int): Number of generations for Genetic Algorithms.
                beam_width (int): Beam width for Beam Search.
                beam_steps (int): Number of steps for Beam Search.

            Returns:
                best_map (dict): The best mapping found across all restarts.
                best_score (float): The score of the best mapping.
            """
            best_map = None
            best_score = -math.inf
            for i in range(restarts):
                #print(f"\n--- Restart {i+1}/{restarts} ---")
                # Create a shuffled initial mapping
                shuffled_map = initial_map.copy()
                letters = list(set(shuffled_map.values()) - set(['_']))
                random.shuffle(letters)
                for sym in shuffled_map:
                    if shuffled_map[sym] != '_':
                        shuffled_map[sym] = letters.pop()
                # Apply Hill-Climbing
                optimized_map, optimized_score = hill_climb(ciphertext, scorer, shuffled_map, hill_iterations, early_stop_threshold=1000)
                #print(f"Restart {i+1}: Hill-Climbing Score = {optimized_score}")
                # Apply Simulated Annealing
                optimized_map, optimized_score = simulated_annealing(ciphertext, scorer, optimized_map, sa_iterations)
                #print(f"Restart {i+1}: Simulated Annealing Score = {optimized_score}")
                # Apply Genetic Algorithms
                optimized_map, optimized_score = genetic_algorithm(ciphertext, scorer, population_size=100, generations=ga_generations, mutation_rate=0.1, elitism=True, elitism_count=5)
                #print(f"Restart {i+1}: Genetic Algorithms Score = {optimized_score}")
                # Apply Beam Search
                optimized_map, optimized_score = beam_search(ciphertext, scorer, optimized_map, beam_width=beam_width, max_steps=beam_steps)
                #print(f"Restart {i+1}: Beam Search Score = {optimized_score}")

                # Update best mapping
                if optimized_score > best_score:
                    best_score = optimized_score
                    best_map = optimized_map

            return best_map, best_score

        # --------------------------------------------------------------------
        # 16. Combined Optimization: Hill-Climbing, Simulated Annealing, Genetic Algorithms, Beam Search
        # --------------------------------------------------------------------
        def optimize_mapping(ciphertext, scorer, initial_map, hill_iterations=10000, sa_iterations=10000, ga_generations=200, beam_width=10, beam_steps=200, restarts=5):
            """
            Optimize the mapping using Hill-Climbing, Simulated Annealing, Genetic Algorithms, Beam Search, and Multiple Restarts.

            Parameters:
                ciphertext (str): The ciphertext to decrypt.
                scorer (Scorer): The scoring object.
                initial_map (dict): The initial mapping.
                hill_iterations (int): Number of iterations for Hill-Climbing.
                sa_iterations (int): Number of iterations for Simulated Annealing.
                ga_generations (int): Number of generations for Genetic Algorithms.
                beam_width (int): Beam width for Beam Search.
                beam_steps (int): Number of steps for Beam Search.
                restarts (int): Number of optimization restarts.

            Returns:
                best_map (dict): The best mapping found.
                best_score (float): The score of the best mapping.
            """
            # Perform multiple restarts
            best_map, best_score = multiple_restarts(
                ciphertext, 
                scorer, 
                initial_map, 
                restarts=restarts, 
                hill_iterations=hill_iterations, 
                sa_iterations=sa_iterations, 
                ga_generations=ga_generations, 
                beam_width=beam_width, 
                beam_steps=beam_steps
            )

            return best_map, best_score

        # --------------------------------------------------------------------
        # 17. Main decryption function
        # --------------------------------------------------------------------
        def decrypt_substitution_cipher(input_ciphertext):
            """
            Decrypt the substitution cipher using frequency analysis and optimization.

            This function:
                - Uses the provided ciphertext.
                - Performs frequency analysis.
                - Applies Hill Climbing, Simulated Annealing, Genetic Algorithms, and Beam Search.
                - Implements multiple restarts.
                - Returns the best decrypted text and the corresponding mapping.
            """
            # Seed the random number generators for reproducibility
            seed_random_generators()

            # Load NLTK corpora
            load_nltk_corpora()

            # Load dictionary and n-gram frequencies
            dictionary = load_dictionary()
            bigram_freq, unseen_prob_bigram = load_bigram_freq()
            trigram_freq, unseen_prob_trigram = load_trigram_freq()

            # Initialize scorer with adjusted weights
            scorer = Scorer(
                dictionary, 
                bigram_freq, 
                trigram_freq, 
                unseen_prob_bigram, 
                unseen_prob_trigram, 
                weights=(20, 10, 5)  # (dict_weight, trigram_weight, bigram_weight)
            )

            # Build initial mapping using frequency analysis
            initial_map = build_initial_mapping(input_ciphertext)
            initial_decoded = decode(input_ciphertext, initial_map)
            initial_score = scorer.score(initial_decoded)
            #print("\n=== Initial Mapping based on Frequency Analysis ===")
            #print_mapping(initial_map)
            #print(f"Initial Decoded Text Score: {initial_score}\n")

            # Perform combined optimization with multiple restarts
            start_time = time.time()
            final_map, final_score = optimize_mapping(
                input_ciphertext, 
                scorer, 
                initial_map, 
                hill_iterations=10000, 
                sa_iterations=10000, 
                ga_generations=200, 
                beam_width=10, 
                beam_steps=200, 
                restarts=5  # Number of restarts for multiple restarts
            )
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Decode with final mapping
            decrypted_text = decode(input_ciphertext, final_map)

            # Prepare deciphered key as a sorted string
            #deciphered_key = ', '.join([f"{sym} -> {final_map[sym]}" for sym in CIPHERTEXT_ALPHABET])
            plaintext_to_cipher = {v: k for k, v in final_map.items()}
            deciphered_key = ''.join([plaintext_to_cipher.get(char, '_') for char in PLAINTEXT_ALPHABET])

            return decrypted_text, deciphered_key

        # --------------------------------------------------------------------
        # 18. Utility: Print the mapping in a readable format
        # --------------------------------------------------------------------
        def print_mapping(mapping):
            """
            Print the ciphertext to plaintext letter mapping in a sorted order.
            """
            sorted_symbols = sorted(CIPHERTEXT_ALPHABET, key=lambda x: CIPHERTEXT_ALPHABET.index(x))
            mapping_str = ', '.join([f"{sym} -> {mapping[sym]}" for sym in sorted_symbols])
            print(mapping_str)

        # --------------------------------------------------------------------
        # Execute the decryption
        # --------------------------------------------------------------------
        decrypted_text, deciphered_key = decrypt_substitution_cipher(ciphertext)

        # Print the required outputs
        print("Ciphertext: " + ciphertext) # Do not change this
        print("Deciphered Plaintext: " + decrypted_text) # Do not change this
        print("Deciphered Key: " + deciphered_key) # Do not change this

        return decrypted_text, deciphered_key # Do not change this
'''
if __name__ == '__main__': # Do not change this
    DecipherText() # Do not change this
'''
if __name__ == '__main__': # Do not change this
    decipherer = DecipherText() # Do not change this
    sample_ciphertext = """1981y, $pp1n1yuux oq@ 2@3s5u1n $p 1981y, 1v y n$s9o2x 19 v$soq yv1y. 1o 1v oq@ v@6@9oq uy27@vo n$s9o2x 5x y2@y, oq@ v@n$98 0$vo 3$3su$sv n$s9o2x, y98 oq@ 0$vo 3$3su$sv 8@0$n2ynx 19 oq@ #$2u8. 5$s98@8 5x oq@ 1981y9 $n@y9 $9 oq@ v$soq, oq@ y2y51y9 v@y $9 oq@ v$soq#@vo, y98 oq@ 5yx $p 5@97yu $9 oq@ v$soq@yvo, 1o vqy2@v uy98 5$28@2v #1oq 3yw1voy9 o$ oq@ #@vo; nq19y, 9@3yu, y98 5qsoy9 o$ oq@ 9$2oq; y98 5y97uy8@vq y98 0xy90y2 o$ oq@ @yvo. 19 oq@ 1981y9 $n@y9, 1981y 1v 19 oq@ 61n191ox $p v21 uy9wy y98 oq@ 0yu816@v; 1ov y98y0y9 y98 91n$5y2 1vuy98v vqy2@ y 0y21o10@ 5$28@2 #1oq oqy1uy98, 0xy90y2 y98 198$9@v1y. 7$$8, 9$# os29 p$2 oq@ v@n$98 3y2o $p oq@ 4s@vo1$9, 7$$8 usnw!"""  # Example ciphertext
    decipherer.decipher(sample_ciphertext)