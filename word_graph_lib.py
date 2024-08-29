import numpy as np
from collections import deque

def get_words(file):
    with open(file, 'r') as f:
        words = f.read().splitlines()
    return words

class WordGraph:
    def __init__(self, words):
        self.words = words
        self.num_words = len(words)
        self.max_len = max(len(word) for word in self.words)
        self.word_graph = self.generate_graph(words)

    def num_edges(self):
        return sum(len(neighbors) for neighbors in self.word_graph.values()) // 2

    def get_neighbors(self, word):
        word_index = self.words.index(word)
        return [self.words[neighbor] for neighbor in self.word_graph[word_index]]

    def get_one_hot(self, word_matrix):
        """
        word_matrix: (num_words, max_len) matrix of words

        Returns a (num_words, max_len, 27) one-hot matrix
        """
        one_hot = np.zeros((word_matrix.shape[0], word_matrix.shape[1], 27), dtype=np.int8)
        one_hot[np.arange(word_matrix.shape[0])[:, None], np.arange(word_matrix.shape[1])[None, :], word_matrix] = 1
        return one_hot

    def get_char_counts(self, word_matrix):
        """
        word_matrix: (num_words, max_len) matrix of words

        Returns a (num_words, 27) matrix of character counts
        """
        one_hot = self.get_one_hot(word_matrix)
        # Sum along the max_len axis
        return np.sum(one_hot, axis=1)

    def generate_graph(self, words):
        word_matrix = np.zeros((self.num_words, self.max_len), dtype=np.int8)
        word_lengths = np.zeros(self.num_words, dtype=np.int8)

        for i, word in enumerate(words):
            word_matrix[i, :len(word)] = [ord(c) for c in word]
            word_lengths[i] = len(word)

        pad_mask = word_matrix == 0

        # Remove capitalization
        word_matrix = word_matrix + 32 * (word_matrix < 91) * (word_matrix > 64)

        # Shift so that 'a' is 1
        word_matrix[~pad_mask] -= ord('a') - 1

        one_hot_word_matrix = self.get_one_hot(word_matrix)
        char_counts_matrix = self.get_char_counts(word_matrix)

        spot_change_pairs = self.get_spot_change_pairs(word_matrix, word_lengths, process_batch_size=500000, max_word_length=None)
        insertion_pairs = self.get_insertion_pairs(word_matrix, word_lengths)
        return self.construct_word_graph(word_matrix, spot_change_pairs, insertion_pairs, word_lengths)

    def get_spot_change_pairs(self, word_matrix, word_lengths, process_batch_size=10000, max_word_length=None):
        spot_change_pairs = {}
        if max_word_length is None:
            max_word_length = max(word_lengths)
        for word_len in range(1, max_word_length + 1):
            print(f"Processing words of length {word_len}")
            word_indices = np.where(word_lengths == word_len)[0]
            print(f"\tNumber of words of length {word_len}: {len(word_indices)}")
            one_hot_word_len_matrix = self.get_one_hot(word_matrix[word_indices])

            # We can only process process_batch_size words at a time due to memory constraints
            spot_change_pairs[word_len] = []
            root_batch_size = process_batch_size // len(word_indices)
            print(f"\tRoot batch size: {root_batch_size} ({root_batch_size * len(word_indices)} word pairs per batch)")
            for start_index in range(0, len(word_indices), root_batch_size):
                end_index = min(start_index + root_batch_size, len(word_indices))
                one_hot_diff_matrix = one_hot_word_len_matrix[start_index:end_index, None] - one_hot_word_len_matrix[None, :]
                char_change_count = np.sum(np.abs(one_hot_diff_matrix), axis=(2, 3))
                spot_change_mask = char_change_count == 2
                spot_change_word_1s, spot_change_word_2s = np.where(spot_change_mask)
                # We must add the start index to the first index
                spot_change_word_1s += start_index
                original_spot_change_word_1s = word_indices[spot_change_word_1s]
                original_spot_change_word_2s = word_indices[spot_change_word_2s]
                # We only add the pairs where the index of the first word is less than the index of the second word
                # This is to avoid reversed duplicates
                # spot_change_pairs[word_len].extend(zip(original_spot_change_word_1s, original_spot_change_word_2s))
                for word_1, word_2 in zip(original_spot_change_word_1s, original_spot_change_word_2s):
                    if word_1 < word_2:
                        spot_change_pairs[word_len].append((word_1, word_2))
        return spot_change_pairs

    def get_insertion_pairs(self, word_matrix, word_lengths, max_word_length=None):
        insertion_pairs = {}
        if max_word_length is None:
            max_word_length = max(word_lengths) 
        for word_len in range(1, max_word_length):
            print(f"Processing insertions from words of length {word_len} to words of length {word_len + 1}")
            root_indices = np.where(word_lengths == word_len)[0]
            root_char_counts = self.get_char_counts(word_matrix[root_indices])

            add_indices = np.where(word_lengths == word_len + 1)[0]
            add_char_counts = self.get_char_counts(word_matrix[add_indices])

            # Pairwise subtract the char counts (And remove the first column, which is the count of empty characters)
            char_count_diff = (add_char_counts[:, None] - root_char_counts[None, :])[:, :, 1:]
            char_count_diff_sum = np.sum(np.abs(char_count_diff), axis=-1)
            candidate_insertion_mask = char_count_diff_sum == 1
            candidate_insertion_indices = np.where(candidate_insertion_mask)
            num_candidate_insertions = len(candidate_insertion_indices[0])
            print(f"\tFound {num_candidate_insertions} candidate insertion pairs")

            # Now we construct two new matrices, one with the root words and one with the added words
            root_word_matrix = np.zeros((num_candidate_insertions, word_len), dtype=np.int8)
            add_word_matrix = np.zeros((num_candidate_insertions, word_len + 1), dtype=np.int8)
            for i, (add_index, root_index) in enumerate(zip(*candidate_insertion_indices)):
                root_word_global_index = root_indices[root_index]
                add_word_global_index = add_indices[add_index]
                root_word = word_matrix[root_word_global_index]
                add_word = word_matrix[add_word_global_index]

                root_word_matrix[i] = root_word[:word_len]
                add_word_matrix[i] = add_word[:word_len + 1]

            # Then we construct the matrix of copies of the added words which has size (num_candidate_insertions, word_len + 1, word_len)
            add_word_matrix_removals = np.zeros((num_candidate_insertions, word_len + 1, word_len), dtype=np.int8)
            for i in range(word_len + 1):
                add_word_matrix_removals[:, i, :] = np.delete(add_word_matrix, i, axis=1)

            # And now we can broadcast compare the root words with the copies of the added words
            matches = (add_word_matrix_removals == root_word_matrix[:, None, :])
            match_mask = np.any(np.all(matches, axis=-1), axis=-1)

            # And now we work backwards from the indices of the matches to the global indices of the pairs
            matched_indices = np.where(match_mask)[0]
            verified_insertion_indices_roots = root_indices[candidate_insertion_indices[1][matched_indices]]
            verified_insertion_indices_adds = add_indices[candidate_insertion_indices[0][matched_indices]]
            verified_insertion_indices = list(zip(verified_insertion_indices_roots, verified_insertion_indices_adds))

            insertion_pairs[word_len] = verified_insertion_indices
            print(f"\tFound {len(verified_insertion_indices)} verified insertion pairs")
        return insertion_pairs

    def construct_word_graph(self, word_matrix, spot_change_pairs, insertion_pairs, word_lengths):
        word_graph = {}
        # Initialize the graph by adding all the words as nodes
        for i in range(len(word_matrix)):
            word_graph[i] = []
        for word_len in spot_change_pairs.keys():
            for word_1, word_2 in spot_change_pairs[word_len]:
                word_graph[word_1].append(word_2)
                word_graph[word_2].append(word_1)
            if word_len < max(word_lengths) - 1:
                for word_1, word_2 in insertion_pairs[word_len]:
                    word_graph[word_1].append(word_2)
                    word_graph[word_2].append(word_1)
        return word_graph

    def bfs(self, start_node, end_node=None):
        """
        start_node: The node to start the search from
        end_node: The node to search for. If None, the search will continue until all nodes are visited and the depth will be the depth of the farthest node.

        returns: The path from the start node to the end node
        """
        start_node = self.words.index(start_node)
        if end_node is not None:
            end_node = self.words.index(end_node)

        visited = np.full(len(self.word_graph), -1, dtype=np.int32)
        queue = deque([(start_node, 0)])
        traversed_nodes = set()

        def recover_path(node):
            path = []
            while node != start_node:
                path.append(self.words[node])
                node = visited[node]
            path.append(self.words[start_node])
            return path[::-1]

        while len(queue) > 0:
            current_node, current_depth = queue.popleft()
            traversed_nodes.add(self.words[current_node])
            if current_node == end_node:
                traversed_nodes.add(self.words[end_node])
                return recover_path(current_node), current_depth, traversed_nodes
            for neighbor in self.word_graph[current_node]:
                if visited[neighbor] == -1:
                    visited[neighbor] = current_node
                    queue.append((neighbor, current_depth + 1))
        if end_node is None:
            # Then we were looking for the farthest node and we found it
            return recover_path(current_node), current_depth, traversed_nodes
        else:
            # Then we didn't find the end node
            return None, None, traversed_nodes

    def find_disconnected_components(self):
        """
        Finds the disconnected components of the graph
        """
        untraversed_nodes = set(self.words)
        components = []
        while len(untraversed_nodes) > 0:
            start_node = untraversed_nodes.pop()
            _, _, traversed_nodes = self.bfs(start_node)
            untraversed_nodes -= traversed_nodes
            components.append(traversed_nodes)
        return components

    def find_diameter(self):
        """
        Uses two BFSs to find the diameter of the graph
        """
        disconnected_components = self.find_disconnected_components()
        longest_path_length = 0
        longest_path = None
        # Order the components by decreasing length
        disconnected_components = sorted(disconnected_components, key=len, reverse=True)

        for component in disconnected_components:
            if len(component) <= longest_path_length:
                # Then it is impossible for this component to have a longer path than the current longest path
                continue
            start_node = component.pop()
            path, _, _ = self.bfs(start_node)
            farthest_node = path[-1]
            path, depth, _ = self.bfs(farthest_node)
            if depth > longest_path_length:
                longest_path_length = depth
                longest_path = path

        return longest_path, longest_path_length

    def to_dot(self, allowed_lengths=None):
        # Now we convert to the DOT format
        keywords = ['NODE', 'EDGE', 'GRAPH', 'DIGRAPH', 'SUBGRAPH', 'STRICT']
        def fix_keyword(w):
            w = w.upper()
            if w in keywords:
                return '_' + w
            return w

        added_edges = set()
        dot_lines = []
        dot_lines.append("graph words {")
        for node in self.word_graph.keys():
            if allowed_lengths is not None and word_lengths[node] not in allowed_lengths:
                continue
            root_word = fix_keyword(self.words[node])
            dot_lines.append(f"\t\"{root_word}\";")
        for node, neighbors in self.word_graph.items():
            if allowed_lengths is not None and word_lengths[node] not in allowed_lengths:
                continue
            root_word = fix_keyword(self.words[node])
            for neighbor in neighbors:
                if allowed_lengths is not None and word_lengths[neighbor] not in allowed_lengths:
                    continue
                if (node, neighbor) in added_edges or (neighbor, node) in added_edges:
                    continue
                added_edges.add((node, neighbor))
                neighbor_word = fix_keyword(self.words[neighbor])
                dot_lines.append(f"\t\"{root_word}\" -- \"{neighbor_word}\";")
        dot_lines.append("}")
        return '\n'.join(dot_lines)