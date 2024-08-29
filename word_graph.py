from word_graph_lib import WordGraph, get_words

def main():
    words = get_words('12dicts_words.txt')

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

    graph = WordGraph(words)
    dot_string = graph.to_dot()
    with open('word_graph.dot', 'w') as f:
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