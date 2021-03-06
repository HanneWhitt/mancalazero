
import tensorflow as tf


class PolicyAndValue(tf.keras.Model):

    '''
    A neural network to take as input:
    - mancala board state vector s
    - mancala board binary legal moves mask (len 6)
    And output: 
    - a policy vector p (len 6) 
    - scalar estimated value v

    Main body is a simple MLP with batch normalisation, effectively produces a board embedding. 
    
    Both policy and value head then have a layer or two to themselves to enable some specialisation in the way they use 
    the shared board embedding. Trying to imitate approach taken in much bigger CNN in DeepMind publications

    Note batchnorm applied AFTER activation function in all cases... Not done in original paper but experiments since seem to 
    support this approach https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
    '''

    def __init__(self, 
                main_body_hidden_layers=[32, 32, 32, 32], 
                board_embedding_dimension=32,
                policy_head_layers=[32, 32],
                value_head_layers=[32, 32],
                main_activation='relu', 
                batchnorm_kwargs={'momentum': 0.1}):

        super(PolicyAndValue, self).__init__()

        self.main_body = MLP(self, 
                board_embedding_dimension,
                hidden_layers=main_body_hidden_layers, 
                main_activation=main_activation, 
                out_activation='linear', 
                batchnorm_kwargs=batchnorm_kwargs)

        self.use_batchnorm = bool(batchnorm_kwargs)
        if self.use_batchnorm:
            self.embedding_batchnorm = tf.keras.layers.BatchNormalization(**batchnorm_kwargs)

        self.policy_head = MLP(self, 
                            6, # 6 possible moves in mancala
                            hidden_layers=policy_head_layers, 
                            main_activation=main_activation, 
                            out_activation='linear', # No output activation yet - apply legal move mask first, then softmax
                            batchnorm_kwargs=batchnorm_kwargs)

        self.policy_softmax = tf.keras.layers.Softmax()
        
        self.value_head = MLP(self, 
                            1, # scalar estimated value
                            hidden_layers=value_head_layers, 
                            main_activation=main_activation, 
                            out_activation='tanh', # We want a value between -1 (certain loss) and 1 (certain win).
                            batchnorm_kwargs=batchnorm_kwargs)


    def call(self, s, legal_moves):

        s = self.main_body(s)

        if self.use_batchnorm:
            s = self.embedding_batchnorm(s)

        # Force the network to return a valid probability distribution with 0 for illegal moves
        p = self.policy_head(s)
        p = tf.multiply(p, legal_moves)
        p = self.policy_softmax(p)

        v = self.value_head(s)

        return p, v


 

class MLP(tf.keras.layers.Layer):

    '''
    Basic MLP with options for batchnorm
    '''

    def __init__(self, 
                output_dim, 
                hidden_layers=[32, 32], 
                main_activation='relu', 
                out_activation='linear', 
                batchnorm_kwargs={'momentum': 0.1}):

        super(MLP, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(n_neurons, activation=main_activation) for n_neurons in hidden_layers]
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=out_activation)
        self.use_batchnorm = bool(batchnorm_kwargs)

        if self.use_batchnorm:
            self.batchnorm_layers = [tf.keras.layers.BatchNormalization(**batchnorm_kwargs) for n_neurons in hidden_layers]


    def call(self, x):
        for l, dense in enumerate(self.hidden_layers):
            x = dense(x)
            if self.use_batchnorm:
                x = self.batchnorm_layers[l](x)
        x = self.output_layer(x)
        return x