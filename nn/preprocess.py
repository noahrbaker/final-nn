# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    labels = np.array(labels)
    pos_indices = np.where(labels)[0]
    neg_indices = np.where(~labels)[0]
    
    # of samples in majority class
    n_samples = max(len(pos_indices), len(neg_indices))
    
    # sample with replacement from minority class
    if len(pos_indices) < len(neg_indices):
        pos_indices = np.random.choice(pos_indices, size=n_samples, replace=True)
    else:
        neg_indices = np.random.choice(neg_indices, size=n_samples, replace=True)
    
    # combine
    all_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(all_indices)
    sampled_seqs = [seqs[i] for i in all_indices]
    sampled_labels = [labels[i] for i in all_indices]
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    valid_nucs = set('ATCG')
    for seq in seq_arr:
        if not set(seq.upper()).issubset(valid_nucs):
            raise ValueError(f"Invalid nucleotides in sequence: {seq}")
    if not seq_arr:
        raise ValueError("Input array is empty")
    if not all(len(seq) == len(seq_arr[0]) for seq in seq_arr):
        raise ValueError("Sequences must be the same length")

    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    seq_len = len(seq_arr[0])
    n_seqs = len(seq_arr)
    
    encodings = np.zeros((n_seqs, seq_len * 4))
    
    for i, seq in enumerate(seq_arr):
        for j, nuc in enumerate(seq.upper()):
            if nuc in mapping:
                encodings[i, j*4:(j+1)*4] = mapping[nuc]
    
    return encodings