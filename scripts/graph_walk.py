from nltk.corpus import wordnet as wn
import random


def try_move(x, movetype):
    """
    Try to make a move. If the specific movetype is impossible, return None
    """
    options = []
    if movetype == "hyponym":
        options = x.hyponyms()
    elif movetype == "hypernym":
        options = x.hypernyms()
    elif movetype == "instance_hypernyms":
        options = x.instance_hypernyms()
    elif movetype == "instance_hyponyms":
        options = x.instance_hyponyms()
    elif movetype == "member_holonyms":
        options = x.member_holonyms()
    elif movetype == "member_meronyms":
        options = x.member_meronyms()

    if len(options) > 0:
        return random.choice(options)
    return None


def make_move(x):
    """
    Make a random move, falling down through move types if a given move is
    not possible. Returns and empty array if no moves are possible.
    """
    m = MOVES.copy()
    random.shuffle(m)
    for move in m:
        new_node = try_move(x, move)
        if new_node is not None:
            return [move, new_node]
    return []


MOVES = [
    "hypernym",
    "hyponym",
    "instance_hypernyms",
    "instance_hypernyms",
    "member_holonyms",
    "member_meronyms",
]

MIN_STEPS = 8
MAX_STEPS = 16


examples = list({i for i in wn.all_synsets()})

walks = []
for synset in examples[:100000]:
    walk = []
    done = False
    step = [synset]
    while len(walk) < MAX_STEPS and len(step) > 0:
        walk.extend(step)
        step = make_move(walk[-1])
    if len(walk) > MIN_STEPS:
        walks.append(walk)

with open("data/walked_paths.txt", "w") as f:
    f.write("\n".join([" ".join([str(j) for j in i]) for i in walks]))
