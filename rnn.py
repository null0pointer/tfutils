from tensorflow.contrib import rnn

DEFAULT_GRU_INTERNAL_SIZE = 512
DEFAULT_DROPOUT_KEEP_RATE = 1.0
DEFAULT_MULTI_GRU_LAYERS = 3

def DropoutGRUCell(size=DEFAULT_GRU_INTERNAL_SIZE, pkeep=DEFAULT_DROPOUT_KEEP_RATE):
    cell = rnn.GRUCell(size)
    cell = rnn.DropoutWrapper(cell, input_keep_prob=pkeep)
    return cell
    
def MultiDropoutGRUCell(size=DEFAULT_GRU_INTERNAL_SIZE, pkeep=DEFAULT_DROPOUT_KEEP_RATE, nlayers=DEFAULT_MULTI_GRU_LAYERS):
    cell = DropoutGRUCell(size=size, pkeep=pkeep)
    cell = rnn.MultiRNNCell([cell] * nlayers, state_is_tuple=False)
    cell = rnn.DropoutWrapper(cell, output_keep_prob=pkeep)
    return cell