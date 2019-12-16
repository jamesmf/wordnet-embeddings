from nltk.corpus import wordnet as wn
import numpy as np
import random


def try_move(x, movetype, walk):
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

    ws = [str(i) for i in walk]
    options = [i for i in options if str(i) not in ws]
    if len(options) > 0:
        return random.choice(options)
    return None


def make_move(walk):
    """
    Make a random move, falling down through move types if a given move is
    not possible. Returns and empty array if no moves are possible.
    """
    x = walk[-1]
    m = MOVES.copy()
    random.shuffle(m)
    for move in m:
        new_node = try_move(x, move, walk)
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

MIN_STEPS = 6
MAX_STEPS = 6


def do_walk(examples, min_len, max_len):
    ind = np.random.randint(0, len(examples))
    synset = examples[ind]
    walk = []
    step = [synset]
    while len(walk) < max_len and len(step) > 0:
        walk.extend(step)
        step = make_move(walk)
    if len(walk) >= min_len:
        return walk
    else:
        return do_walk(examples, min_len, max_len)
