import numpy as np
from Bio import SeqIO
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
from einops import rearrange
np.random.seed(1)


def onehot(seq):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'unk': [0, 0, 0, 0]}
    encoded = np.zeros(shape=(len(seq), 4))
    for i, nt in enumerate(seq):
        if nt in ['A', 'C', 'G', 'T']:
            encoded[i, :] = code[nt]
        else:
            encoded[i, :] = code['unk']
    return encoded


def shuffle_several_times(seqs: np.array, reps=10):
  seqs = np.array(seqs)
  assert len(seqs.shape) == 3
  sep_shuffled_seqs = np.array([dinuc_shuffle(s, num_shufs=reps) for s in seqs])
  shuffle_out = rearrange(sep_shuffled_seqs, "b r l n -> (b r) l n")
  return shuffle_out


def prepare_seqs(fasta_file):
    seqs = []
    seq_ids = []
    for rec in SeqIO.parse(fasta_file, format='fasta'):
        upstream, downstream = str(rec.seq).split('BREAK')
        onehot_seq = np.concatenate([onehot(upstream), np.zeros(shape=(20, 4)), onehot(downstream)])
        seqs.append(onehot_seq)
        seq_ids.append(rec.id)
    seqs = np.array(seqs)
    return seqs, seq_ids


def compute_scores(onehot_data, keras_model):
    dinuc_shuff_explainer = shap.DeepExplainer(model=(keras_model.input, keras_model.output[:, 0]),
                                               data=shuffle_several_times)
    raw_shap_explanations = dinuc_shuff_explainer.shap_values(onehot_data, check_additivity=False)
    dinuc_shuff_explanations = (np.sum(raw_shap_explanations, axis=-1)[:, :, None] * onehot_data)

    return dinuc_shuff_explanations


