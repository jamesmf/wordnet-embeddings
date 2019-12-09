import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

EMBEDDING_DIM = 64
MAX_WALK_LEN = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 64

with open("data/walked_paths.txt", "r") as f:
    walks = f.readlines()

vocab = set()
for walk in walks:
    for node in walk.strip().split(" "):
        vocab.add(node)
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
    lstm = krs.layers.LSTM(EMBEDDING_DIM)
    # vars for loss
    output_weights = tf.Variable(tf.random.normal([vocab_len, EMBEDDING_DIM]))
    output_biases = tf.Variable(tf.zeros([vocab_len]))

    output_ph = krs.backend.placeholder((None, 1))


def get_embedding(x):
    with tf.device("/GPU:0"):
        # Lookup the corresponding embedding vectors for each sample in X.
        x_embed = embedding(x)
        x_out = lstm(x_embed)
        return x_out


def nce_loss(x_inp, y):
    with tf.device("/GPU:0"):
        # Compute the average NCE loss for the batch.
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=output_weights,
                biases=output_biases,
                labels=y,
                inputs=x_inp,
                num_sampled=5,
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
        embedding_norm = embedding / tf.sqrt(
            tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32
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
            loss = nce_loss(emb, y)

        # Compute gradients.
        to_diff = embedding.weights + lstm.weights + [output_weights, output_biases]
        gradients = g.gradient(loss, to_diff)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, to_diff))


def next_batch(data, batch_size, ind):
    size = np.min([batch_size, len(data)])
    labels = np.zeros((size, 1))
    inps = np.zeros((size, MAX_WALK_LEN))
    for n, d in enumerate(data[ind : ind + size]):
        example = []
        walk = d.strip().split(" ")
        # see how far we can start
        num_nodes = len(walk) - 1
        start_max = num_nodes - MAX_WALK_LEN
        start = np.random.randint(0, max(start_max, 0))
        for node in walk[start : start + MAX_WALK_LEN]:
            example.append(vocab[node])
        seq = example[-MAX_WALK_LEN - 1 : -1]
        label = example[-1]
        seq_len = len(seq)
        offset = MAX_WALK_LEN - seq_len
        labels[n] = label
        inps[n, offset:] = np.array(seq)
    return inps, labels


ind = 0
for step in range(1, 1000000):
    batch_x, batch_y = next_batch(walks, BATCH_SIZE, ind)
    ind += batch_x.shape[0]
    ind = ind % len(walks)
    run_optimization(batch_x, batch_y)

    if step % 1000 == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))


w = embedding.weights[0].numpy()
with open("data/asvec.vec", "w") as f:
    f.write(str(w.shape[0] - 1) + " " + str(EMBEDDING_DIM) + "\n")
    f.write(
        "\n".join(
            [
                reverse[n] + " " + " ".join([str(i) for i in w[n]])
                for n in range(1, w.shape[0])
            ]
        )
    )
