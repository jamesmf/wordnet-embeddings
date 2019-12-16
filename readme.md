## Wordnet Embeddings

### Installation

`docker build -t wordnet-embeddings .`

This package uses tensorflow v2.

### Learning embeddings

This project continuously randomly walks a graph of all Synsets in English WordNet. It then uses an LSTM over the path to predict the final node in the path. The type of relation is encoded as its own step.

An example path might look like

```
"["Synset('computerized_tomography.n.01')", 'hypernym', "Synset('x-raying.n.01')", 'hypernym', "Synset('imaging.n.02')", 'hypernym']" -----> Synset('pictorial_representation.n.01')
```

The model is trained using `sampled_softmax_loss`, though in reality there's no need to subtract `log(Q(y|x))` since we aren't using any frequency information.

### Evaluating embeddings

As the model trains on these random walks, it evaluates other random walks. It predicts a vector at the last timestep of a given walk, then finds the most cosine-similar Synset in the output embedding weights.
```
"["Synset('computerized_tomography.n.01')", 'hypernym', "Synset('x-raying.n.01')", 'hypernym', "Synset('imaging.n.02')", 'hypernym']" ------> Synset('pictorial_representation.n.01')
 nearest neighbors:
 Synset('pictorial_representation.n.01'),
 Synset('butterfly_bush.n.01'),
 Synset('nectariferous.a.01'),
 Synset('min.n.02'),
 Synset('sonography.n.01'),
 Synset('x-raying.n.01'),



 "["Synset('neva.n.01')", 'instance_hypernyms', "Synset('river.n.01')", 'hypernym', "Synset('stream.n.01')", 'hypernym']"  ------> Synset('body_of_water.n.01')
 nearest neighbors:
 Synset('brook.n.01'),
 Synset('body_of_water.n.01'),
 Synset('headstream.n.01'),
 Synset('branch.n.05'),
 Synset('eparchial.a.01'),
 Synset('pipturus.n.01'),

 ```
