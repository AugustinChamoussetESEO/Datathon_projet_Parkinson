from __future__ import unicode_literals, print_function, division

import torch

from main import evaluate, Language
from models import AttnDecoderRNN, EncoderRNN

SOS_token = 0
EOS_token = 1
SEP_token = 2

if __name__ == '__main__':
    hidden_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package = torch.load('out/checkpoints/best_model.pkl', map_location=device)
    max_length = package['max_length']
    language = package['language']

    encoder = EncoderRNN(language.n_words, hidden_size, device=device).to(device)
    decoder = AttnDecoderRNN(hidden_size, language.n_words, device, max_length, dropout_p=0.1).to(device)
    encoder.load_state_dict(package['encoder_state_dict'])
    decoder.load_state_dict(package['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    test_input = "Bonjour, comment allez-vous aujourd'hui, <NAME>? SEP comment tu t'appelles"
    test_output = evaluate(encoder, decoder, language, test_input, max_length, device)[0]
    del test_output[-1]
    test_output = language.words_to_sentence(test_output)

    print("Input: " + test_input)
    print("Output: " + test_output)


