from __future__ import unicode_literals, print_function, division

import sys
import json
import random
from io import open

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from edith.models import AttnDecoderRNN, EncoderRNN

from unidecode import unidecode

SOS_token = 0
EOS_token = 1
SEP_token = 2


class Language:
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1, "SEP": 2}
        self.word2count = {"SEP": 0}  # doesn't provide the right count
        self.index2word = {0: "SOS", 1: "EOS", 2: "SEP"}
        self.n_words = 3  # Count SOS, EOS and SEP

        self.right_substitutions = [
            [".", " . "],
            ["?", " ? "],
            ["'", " ' "],
            [",", " , "],
            ["-", " - "],
            ["!", " ! "],
            ['"', ' " ']
        ]

        self.left_substitutions = [
            [" . ", ". "],
            [" .", "."],
            [" ? ", "? "],
            [" ?", "?"],
            [" ' ", "'"],
            [" , ", ", "],
            [" - 1", " -1"],
            [" - 2", " -2"],
            [" - 3", " -3"],
            [" - 4", " -4"],
            [" - 5", " -2"],
            [" - 6", " -6"],
            [" - 7", " -7"],
            [" - 8", " -8"],
            [" - 9", " -9"],
            [" - ", "-"],
            [" ! ", "! "],
            [" !", "!"]
        ]

    def add_sentence(self, sentence):
        words = self.sentence_to_words(sentence)
        for word in words:
            self.add_word(word)
        return len(words)

    def sentence_to_words(self, sentence):
        for substitution in self.right_substitutions:
            sentence = sentence.replace(substitution[0], substitution[1])

        return list(filter(None, sentence.split(' ')))

    def words_to_sentence(self, words):
        sentence = " ".join(words)
        for substitution in self.left_substitutions:
            sentence = sentence.replace(substitution[0], substitution[1])
        return sentence

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def tensor_from_sentence(self, sentence, device, max_length):
        indexes = []
        for word in self.sentence_to_words(sentence):  # Ignore unknown words
            if word in self.word2index:
                indexes.append(self.word2index[word])

        indexes.append(EOS_token)
        indexes = indexes[-max_length:]
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,
          teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder, language, sentence, max_length, device):
    with torch.no_grad():
        input_tensor = language.tensor_from_sentence(sentence, device, max_length=max_length)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(language.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def load_file(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        conversations = json.load(file)

    input_data = []
    for conversation in conversations:
        lower_conversation = [x.lower() if i % 2 == 1 else x for i, x in enumerate(conversation)]
        upper_conversation = [x.upper() if i % 2 == 1 else x for i, x in enumerate(conversation)]
        ascii_conversation = [unidecode(x) if i % 2 == 1 else x for i, x in enumerate(conversation)]
        ascii_lower_conversation = [unidecode(x.lower()) if i % 2 == 1 else x for i, x in enumerate(conversation)]
        ascii_upper_conversation = [unidecode(x.upper()) if i % 2 == 1 else x for i, x in enumerate(conversation)]

        for idx in range(1, len(conversation)):
            if idx % 2 == 0:
                input_data.append([" SEP ".join(conversation[:idx]), conversation[idx]])
                input_data.append([" SEP ".join(lower_conversation[:idx]), lower_conversation[idx]])
                input_data.append([" SEP ".join(upper_conversation[:idx]), upper_conversation[idx]])
                input_data.append([" SEP ".join(ascii_conversation[:idx]), ascii_conversation[idx]])
                input_data.append([" SEP ".join(ascii_lower_conversation[:idx]), ascii_lower_conversation[idx]])
                input_data.append([" SEP ".join(ascii_upper_conversation[:idx]), ascii_upper_conversation[idx]])

    input_data = list(set(tuple(sublist) for sublist in input_data))  # Keep only unique items
    input_data = [list(x) for x in input_data]  # Convert it back to list of lists
    return input_data


if __name__ == '__main__':
    epochs = 100
    max_length = 1
    hidden_size = 256
    learning_rate = 0.01
    max_length = 128  # In words

    data = load_file('data/conv.json')
    #data.extend(load_file('data/conversations_2.json'))
    #data.extend(load_file('data/math_operations.json'))

    language = Language()
    for in_data in data:
        language.add_sentence(in_data[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(language.n_words, hidden_size, device=device).to(device)
    decoder = AttnDecoderRNN(hidden_size, language.n_words, device, max_length, dropout_p=0.1).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5, min_lr=1e-6)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5, min_lr=1e-6)
    criterion = nn.NLLLoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        train_losses = []
        pbar = tqdm(unit="batch", file=sys.stdout, total=len(data))
        pbar.set_description("Epoch %d/%d, Lr %f" % (epoch, epochs, encoder_optimizer.param_groups[0]['lr']))
        random.shuffle(data)
        for idx, sample in enumerate(data):
            input_tensor = language.tensor_from_sentence(sample[0], device, max_length)
            target_tensor = language.tensor_from_sentence(sample[1], device, max_length)
            loss_ = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            train_losses.append(loss_)
            pbar.set_postfix({"train_loss": sum(train_losses) / len(train_losses)})
            pbar.update(1)

        final_loss = sum(train_losses) / len(train_losses)
        encoder_scheduler.step(final_loss)
        decoder_scheduler.step(final_loss)

        pbar.close()

        # Save the best model
        if final_loss < best_loss:
            best_loss = final_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'language': language,
                'max_length': max_length
            }, './out/checkpoints/best_model.pkl')
