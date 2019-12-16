import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
import graph_walk

EMBEDDING_DIM = 128
MAX_WALK_LEN = 6
LEARNING_RATE = 0.01
BATCH_SIZE = 64

# with open("data/walked_paths.txt", "r") as f:
#     walks = f.readlines()
synsets = list({i for i in graph_walk.wn.all_synsets()})
vocab = set()
for s in synsets:
    vocab.add(str(s).strip())
for t in graph_walk.MOVES:
    vocab.add(t)
labels = sorted(vocab)
vocab = {i: n + 1 for n, i in enumerate(labels)}
reverse = {v: k for k, v in vocab.items()}
reverse[0] = ""
# add one for the null/masked value
vocab_len = len(vocab) + 1


# model.add()


with tf.device("/GPU:0"):
    # embedding layer and lstm
    embedding = krs.layers.Embedding(vocab_len, EMBEDDING_DIM)
    lstm = krs.layers.LSTM(EMBEDDING_DIM, activation="tanh")
    # vars for loss
    output_weights = tf.Variable(tf.random.normal([vocab_len, EMBEDDING_DIM]))
    output_biases = tf.Variable(tf.zeros([vocab_len]))


def get_embedding(x):
    with tf.device("/GPU:0"):
        # Lookup the corresponding embedding vectors for each sample in X.
        x_embed = embedding(x)
        x_out = lstm(x_embed)
        return x_out


def get_loss(x_inp, y):
    with tf.device("/GPU:0"):
        # Compute the average NCE loss for the batch.
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=output_weights,
                biases=output_biases,
                labels=y,
                inputs=x_inp,
                num_sampled=256,
                num_classes=vocab_len,
                num_true=1,
            )
        )
        return loss


# Evaluation.
def evaluate(x_embed):
    with tf.device("/GPU:0"):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embedding_norm = output_weights / tf.sqrt(
            tf.reduce_sum(tf.square(output_weights), 1, keepdims=True), tf.float32
        )
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
        return cosine_sim_op


# Define the optimizer.
optimizer = tf.optimizers.SGD(LEARNING_RATE)


def run_optimization(x, y):
    y = tf.convert_to_tensor(y)
    with tf.device("/GPU:0"):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = get_loss(emb, y)

        # Compute gradients.
        to_diff = embedding.weights + lstm.weights + [output_weights, output_biases]
        gradients = g.gradient(loss, to_diff)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, to_diff))


# def next_batch(data, batch_size, ind):
#     size = np.min([batch_size, len(data)])
#     labels = np.zeros((size, 1))
#     inps = np.zeros((size, MAX_WALK_LEN))
#     for n, d in enumerate(data[ind : ind + size]):
#         example = []
#         walk = d.strip().split(" ")
#         # see how far we can start
#         num_nodes = len(walk) - 1
#         start_max = num_nodes - MAX_WALK_LEN
#         start_max -= start_max % 2
#         start = np.random.randint(0, max(start_max, 0))
#         start -= start % 2
#         for node in walk[start : start + MAX_WALK_LEN + 1]:
#             example.append(vocab[node])
#         seq = example[-MAX_WALK_LEN - 1 : -1]
#         label = example[-1]
#         seq_len = len(seq)
#         offset = MAX_WALK_LEN - seq_len
#         labels[n] = label
#         inps[n, offset:] = np.array(seq)
#     return inps, labels


def next_batch(synsets, batch_size):
    inps = []
    labels = []
    while len(inps) < batch_size:
        walk = graph_walk.do_walk(synsets, MAX_WALK_LEN + 1, MAX_WALK_LEN + 1)
        inp = walk[:MAX_WALK_LEN]
        label = walk[MAX_WALK_LEN]
        inps.append([vocab[str(val)] for val in inp])
        labels.append(vocab[str(label)])
    return np.array(inps), np.array(labels).reshape(-1, 1)


for step in range(1, 1000000):
    batch_x, batch_y = next_batch(synsets, BATCH_SIZE)
    run_optimization(batch_x, batch_y)

    if step % 1000 == 0 or step == 1:
        loss = get_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))

    if step % 5000 == 0:
        print("Evaluation on training batch...")
        batch_x, batch_y = next_batch(synsets, BATCH_SIZE)
        sim = evaluate(get_embedding(batch_x)).numpy()
        for i in range(5):
            top_k = 6  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[:top_k]
            log_str = (
                '\n\n\n"%s" nearest neighbors:' % str([reverse[j] for j in batch_x[i]])
                + " "
                + str(reverse[int(batch_y[i])])
            )
            for k in range(top_k):
                log_str = "\n%s %s,\n" % (log_str, reverse[nearest[k]])
            print(log_str)


w = embedding.weights[0].numpy()
w_o = output_weights.numpy()
with open("data/asvec.vec", "w") as f:
    f.write(str(w.shape[0] - 1) + " " + str(2 * EMBEDDING_DIM) + "\n")
    f.write(
        "\n".join(
            [
                reverse[n]
                + " "
                + " ".join([str(i) for i in w[n]])
                + " "
                + " ".join([str(i) for i in w_o[n]])
                for n in range(1, w.shape[0])
            ]
        )
    )
