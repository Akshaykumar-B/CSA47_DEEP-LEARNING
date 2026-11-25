"""Train a small word-level LSTM encoder-decoder on parallel phrase pairs."""

import itertools

import torch
from torch import nn


torch.manual_seed(65)
torch.set_num_threads(1)
colors = {
    "red": "rouge",
    "blue": "bleu",
    "green": "vert",
    "yellow": "jaune",
}
fruits = {
    "apple": "pomme",
    "pear": "poire",
    "plum": "prune",
    "peach": "peche",
}
all_pairs = list(itertools.product(colors, fruits))
test_pairs = {("red", "apple"), ("blue", "pear"), ("green", "peach"), ("yellow", "plum")}
train_pairs = [pair for pair in all_pairs if pair not in test_pairs]
source_vocab = {word: index for index, word in enumerate(
    ["<pad>", "i", "like", *colors.keys(), *fruits.keys()]
)}
target_words = ["<pad>", "<sos>", "<eos>", "je", "aime", *fruits.values(), *colors.values()]
target_vocab = {word: index for index, word in enumerate(dict.fromkeys(target_words))}
target_token = {index: word for word, index in target_vocab.items()}


def source_ids(pair):
    color, fruit = pair
    return [source_vocab[word] for word in ("i", "like", color, fruit)]


def target_ids(pair):
    color, fruit = pair
    words = ("je", "aime", fruits[fruit], colors[color])
    return [target_vocab["<sos>"], *[target_vocab[word] for word in words]], [
        *[target_vocab[word] for word in words], target_vocab["<eos>"]
    ]


x_train = torch.tensor([source_ids(pair) for pair in train_pairs])
decoder_inputs, decoder_targets = zip(*(target_ids(pair) for pair in train_pairs))
decoder_inputs = torch.tensor(decoder_inputs)
decoder_targets = torch.tensor(decoder_targets)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_embedding = nn.Embedding(len(source_vocab), 24)
        self.encoder = nn.LSTM(24, 48, batch_first=True)
        self.target_embedding = nn.Embedding(len(target_vocab), 24)
        self.decoder = nn.LSTM(24, 48, batch_first=True)
        self.output = nn.Linear(96, len(target_vocab))

    def forward(self, source, decoder_input):
        encoded, state = self.encoder(self.source_embedding(source))
        decoded, _ = self.decoder(self.target_embedding(decoder_input), state)
        attention = torch.softmax(torch.bmm(decoded, encoded.transpose(1, 2)), dim=-1)
        context = torch.bmm(attention, encoded)
        return self.output(torch.cat([decoded, context], dim=-1))

    def translate(self, source, max_length=6):
        encoded, state = self.encoder(self.source_embedding(source.unsqueeze(0)))
        token = torch.tensor([[target_vocab["<sos>"]]])
        output_words = []
        for _ in range(max_length):
            decoded, state = self.decoder(self.target_embedding(token), state)
            attention = torch.softmax(
                torch.bmm(decoded[:, -1:], encoded.transpose(1, 2)), dim=-1
            )
            context = torch.bmm(attention, encoded)
            token = self.output(torch.cat([decoded[:, -1:], context], dim=-1)).argmax(
                -1
            )
            word = target_token[token.item()]
            if word == "<eos>":
                break
            output_words.append(word)
        return output_words


model = EncoderDecoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
for epoch in range(300):
    optimizer.zero_grad()
    logits = model(x_train, decoder_inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, len(target_vocab)), decoder_targets.reshape(-1))
    loss.backward()
    optimizer.step()

correct = 0
for pair in sorted(test_pairs):
    expected = ["je", "aime", fruits[pair[1]], colors[pair[0]]]
    predicted = model.translate(torch.tensor(source_ids(pair)))
    correct += predicted == expected
    print(f"i like {pair[0]} {pair[1]} -> {' '.join(predicted)} (expected: {' '.join(expected)})")
print(f"Held-out exact-match accuracy: {correct / len(test_pairs):.3f}")
print("This tiny parallel corpus demonstrates the model; it is not a production translator.")
