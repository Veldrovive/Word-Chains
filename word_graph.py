from word_graph_lib import WordGraph, get_words, open_ipa_dict_representations_map, open_cmudict_representations_map

def use_IPA_rep(words):
    whitelisted_words = set(words)
    ipa_dict_representations_map = open_ipa_dict_representations_map(whitelisted_words)
    ipa_words = set(ipa_dict_representations_map.keys())
    missed_words = whitelisted_words - ipa_words
    print(f"Found {len(ipa_words)} words in the IPA dictionary, missed {len(missed_words)} words")

    # Filter down the words and representations to only the ones we have in the IPA dictionary
    ipa_words = []
    ipa_representations = []
    for word, representations in ipa_dict_representations_map.items():
        for representation in representations:
            ipa_words.append(word)  # Duplicate words are allowed as long as they have different representations
            ipa_representations.append(representation)

    words = ipa_words
    representations = ipa_representations

    def label_rep_function(word, representation):
        return f"{word} ({''.join(representation)})"

    return words, representations, label_rep_function

def use_CMU_rep(words):
    whitelisted_words = set(words)
    cmudict_representations_map = open_cmudict_representations_map(whitelisted_words)
    cmudict_words = set(cmudict_representations_map.keys())
    missed_words = whitelisted_words - cmudict_words
    print(f"Found {len(cmudict_words)} words in the CMU dictionary, missed {len(missed_words)} words")

    # Filter down the words and representations to only the ones we have in the CMU dictionary
    cmudict_words = []
    cmudict_representations = []
    for word, representations in cmudict_representations_map.items():
        for representation in representations:
            cmudict_words.append(word)  # Duplicate words are allowed as long as they have different representations
            cmudict_representations.append(representation)

    words = cmudict_words
    representations = cmudict_representations

    def label_rep_function(word, representation):
        return f"{word} ({' '.join(representation)})"

    return words, representations, label_rep_function

def main():
    words = get_words('12dicts_words.txt')
    representations = None
    label_rep_function = None
    file_postfix = ""

    try:
        representation_type = input('Choose the word representation:\n1 - None (The words themselves)\n2 - International Phonetic Alphabet\n3 - Carnegie Mellon University Pronouncing Dictionary\n')
        if representation_type == '1':
            pass
        elif representation_type == '2':
            words, representations, label_rep_function = use_IPA_rep(words)
            file_postfix = "_ipa"
        elif representation_type == '3':
            words, representations, label_rep_function = use_CMU_rep(words)
            file_postfix = "_cmu"
        else:
            raise ValueError
    except ValueError:
        print('Invalid input')
        return

    try:
        max_word_length = input('Enter the maximum length of the words used to build the graph (leave blank for no limit): ')
        if max_word_length:
            max_word_length = int(max_word_length)
            if max_word_length < 1:
                raise ValueError
            words = [word for word in words if len(word) <= max_word_length]
    except ValueError:
        print('Invalid input')
        return

    try:
        min_component_size = input('Enter the minimum size of the connected components used when exporting the graph (leave blank for 10): ')
        if min_component_size:
            min_component_size = int(min_component_size)
            if min_component_size < 1:
                raise ValueError
        else:
            min_component_size = 10
    except ValueError:
        print('Invalid input')
        return

    graph = WordGraph(words, representations, label_rep_function)
    dot_string = graph.to_dot(min_component_size=min_component_size)
    file_name = f'word_graph{file_postfix}.dot'
    with open(file_name, 'w') as f:
        f.write(dot_string)

    print(f"\n\n************")
    print(f"Graph built with {len(words)} words and {graph.num_edges()} edges")
    disconnected_components = graph.find_disconnected_components()
    num_disconnected_components = len(disconnected_components)
    num_disconnected_components_over_2 = len([component for component in disconnected_components if len(component) > 2])
    num_disconnected_components_over_5 = len([component for component in disconnected_components if len(component) > 5])
    print(f"\nGraph has {num_disconnected_components} disconnected components.")
    print(f"Graph has {num_disconnected_components_over_2} disconnected components with more than 2 words.")
    print(f"Graph has {num_disconnected_components_over_5} disconnected components with more than 5 words.")
    longest_path, longest_path_length = graph.find_diameter()
    longest_path = graph.path_to_label(longest_path)
    print(f"\nLongest path in the graph has length {longest_path_length} and is:")
    print(' -> '.join(longest_path))
    print(f"************\n\n")

    def shortest_path_loop():
        print(f"\n**************\nEnter two words to find the shortest path between them.")
        while True:
            start_word = input('Enter the start word (leave blank to exit): ')
            if not start_word:
                break
            if start_word not in graph.words:
                print('Word not found in the graph')
                continue
            end_word = input('Enter the end word: ')
            if end_word not in graph.words:
                print('Word not found in the graph')
                continue
            shortest_path, shortest_path_length, _ = graph.bfs(start_word, end_word)
            shortest_path = graph.path_to_label(shortest_path)
            if shortest_path:
                print(f"\nShortest path between {start_word} and {end_word} has length {shortest_path_length} and is:")
                print(' -> '.join(shortest_path))
            else:
                print(f"\nNo path found between {start_word} and {end_word}")
            print(f"************\n\n")

    def get_neighbors_loop():
        print(f"\n************\nEnter a word to see its neighbors.")
        while True:
            word = input('Enter the word (leave blank to exit): ')
            if not word:
                break
            if word not in graph.words:
                print('Word not found in the graph')
                continue
            neighbors = graph.get_neighbors(word)
            print(f"\nNeighbors of {word}:")
            print(', '.join(neighbors))
            print(f"************\n\n")

    while True:
        option = input('Choose an option:\n1 - Find shortest path between two words\n2 - Get neighbors of a word\n3 - Exit\n')
        if option == '1':
            shortest_path_loop()
        elif option == '2':
            get_neighbors_loop()
        elif option == '3':
            break
        else:
            print('Invalid option')

if __name__ == '__main__':
    main()