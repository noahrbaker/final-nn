import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs


@pytest.fixture
def simple_nn():
    arch = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    return NeuralNetwork(
        nn_arch=arch,
        lr=0.1,
        seed=42,
        batch_size=2,
        epochs=1,
        loss_function='bce'
    )


def test_single_forward():
    # simple_nn = simple_nn()
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    b = np.array([[0.1], [0.2]])
    A_prev = np.array([[1.0], [2.0]])
    expected_Z = np.array([[0.6], [1.3]])
    expected_A = np.array([[0.6], [1.3]])

    # test ReLU
    A_relu, Z_relu = simple_nn._single_forward(W, b, A_prev, 'relu')
    np.testing.assert_array_almost_equal(Z_relu, expected_Z)
    np.testing.assert_array_almost_equal(A_relu, expected_A)

    # test Sigmoid
    A_sig, _ = simple_nn._single_forward(W, b, A_prev, 'sigmoid')
    expected_A_sig = 1 / (1 + np.exp(-expected_Z))
    np.testing.assert_array_almost_equal(A_sig, expected_A_sig)

def test_forward():
    # simple_nn = simple_nn()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    output, cache = simple_nn.forward(X)
    
    assert all(f'A{i}' in cache for i in range(3))
    assert all(f'Z{i}' in cache for i in range(1, 3))
    assert output.shape == (1, 2)  # Final layer output
    assert cache['A0'].shape == (2, 2)  # Input layer

def test_single_backprop():
    # simple_nn = simple_nn()
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    b = np.array([[0.1], [0.2]])
    Z = np.array([[0.5], [1.1]])
    A_prev = np.array([[1.0], [2.0]])
    dA = np.array([[0.1], [0.2]])
    dA_prev, dW, db = simple_nn._single_backprop(
        W, b, Z, A_prev, dA, 'relu'
    )
    
    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

def test_predict():
    # simple_nn = simple_nn()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = simple_nn.predict(X)

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))

def test_binary_cross_entropy():
    # simple_nn = simple_nn()
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.3], [0.2, 0.8]])
    loss = simple_nn._binary_cross_entropy(y, y_hat)

    assert isinstance(loss, float)
    assert loss >= 0

def test_binary_cross_entropy_backprop():
    # simple_nn = simple_nn()
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.3], [0.2, 0.8]])
    dA = simple_nn._binary_cross_entropy_backprop(y, y_hat)

    assert dA.shape == y.shape

def test_mean_squared_error():
    # simple_nn = simple_nn()
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_hat = np.array([[0.9, 0.1], [0.1, 0.9]])
    loss = simple_nn._mean_squared_error(y, y_hat)

    assert isinstance(loss, float)
    assert loss >= 0

def test_mean_squared_error_backprop():
    # simple_nn = simple_nn()
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_hat = np.array([[0.9, 0.1], [0.1, 0.9]])
    dA = simple_nn._mean_squared_error_backprop(y, y_hat)

    assert dA.shape == y.shape


def test_sample_seqs():
    seqs = ['ATCG', 'GCTA', 'TTAA', 'GGCC']
    labels = [True, False, True, False]
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    
    assert len(sampled_seqs) == len(sampled_labels)
    assert len(sampled_seqs) >= len(seqs)
    assert all(len(seq) == len(seqs[0]) for seq in sampled_seqs)
    
    pos_count = sum(sampled_labels)
    neg_count = len(sampled_labels) - pos_count
    
    assert pos_count == neg_count

def test_one_hot_encode_seqs():
    seqs = ['ATCG', 'GCTA']
    encodings = one_hot_encode_seqs(seqs)
    
    assert encodings.shape == (2, 16)
    
    expected_first_seq = np.array([
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    ])
    np.testing.assert_array_equal(encodings[0], expected_first_seq)