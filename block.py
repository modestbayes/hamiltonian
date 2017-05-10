from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Dropout, merge
from keras.optimizers import SGD, Adam

def build(input_dim, hidden_dim, output_dim):
    """Builds a neural network for gradient approximation.
    
    The neural network consists of individual blocks for partial gradients.
        
    Args:
        input_dim: int for input dimension
        hidden_dim: list of int for multiple hidden layers
        output_dim: list of int for multiple output layers
        
    Returns:
        A complied Keras model.
    """
    x = Input(shape=(input_dim,))
    
    hidden = []
    for s in hidden_dim:
        hidden.append(Dense(s, activation='relu', bias=True)(x))
        
    output = []
    for i, h in enumerate(hidden):
        output.append(Dense(output_dim[i], activation='linear', bias=True)(h))
        
    if len(output) == 1:
        y = output[0]
    else:
        y = merge(output, mode='concat', concat_axis=1)
    
    model = Model(x, y)
    model.summary()
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return(model)