# Word Chains
Builds off the work of [CodeParade](https://github.com/HackerPoet/WordChainSolver), but extends the concept of adjacency to include pairs with a single insertion or deletion as well as the point swaps.

### Running the script
Install numpy as a dependency: `pip install numpy`

Run the command line utility: `python word_graph.py`

This will prompt you to input the max word length you would like to have in the graph. A max length of 3 will complete almost instantly. 5 will take a few seconds. Pressing enter to process all words will take around 5 minutes on a good computer. I make no claims of this being an especially efficient way to compute this graph, just that it works.

Once it has built the graph, it will output some summary information. Since it is a bit of a hassle to download and run, I will give the completed summary here:
```
Graph built with 64662 words and 86115 edges

Graph has 26505 disconnected components.
Graph has 3024 disconnected components with more than 2 words.
Graph has 563 disconnected components with more than 5 words.

Longest path in the graph has length 42 and is:
hammerings -> hammering -> hampering -> pampering -> papering -> capering -> catering -> cantering -> bantering -> battering -> bettering -> fettering -> festering -> pestering -> petering -> peering -> peeing -> seeing -> sewing -> swing -> sing -> sine -> mine -> mire -> mere -> metre -> metred -> metered -> petered -> pestered -> festered -> fettered -> bettered -> battered -> bantered -> cantered -> catered -> capered -> papered -> pampered -> hampered -> hammered -> yammered
```
We can see that most of the disconnected components consist of a word by itself or a word and its plural. However, there are a substantial number of components with multiple words in their own cluster.

The "-ings" are still the bane of the graph. Although they are connected to the larger graph, it is very difficult to go between an ing and anything else. I find the largest path very interesting because the word root often remains the same on the way from "hammerings" to "yammered", but it changes from "-ing" to "-ed".

### Visualizing the graph
For detailed instructions, reference [CodeParade's README](https://github.com/HackerPoet/WordChainSolver?tab=readme-ov-file#viewing-the-graphs). I have included a built graph at [/word_graph.dot](/word_graph.dot).
