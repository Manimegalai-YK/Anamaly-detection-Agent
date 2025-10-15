import numpy as np

# Load frames/features from file (update filename/path accordingly)
frames = np.load('frames.npy')  
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10
sequences = create_sequences(frames, seq_length)
print('Sequences shape:', sequences.shape)
