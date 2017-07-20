import numpy as np

from keras import backend as K
from keras import activations
from keras.layers import Recurrent
from keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from keras.engine import InputSpec
import tensorflow as tf
from holographic_memory import complex_mult

class PredNet(Recurrent):
    '''PredNet architecture - Lotter 2016.
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, but should not be confused with returning all of the layers of the model
            For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                For instance, to return the features of the LSTM "representational" units in the lowest layer, output_mode should be specificied as 'R0'.
                The possible unit types are 'R', 'Ahat', 'A', and 'E' corresponding to the 'representation', 'prediction', 'target', and 'error' units respectively.
        extrap_start_time: time step for which model will start extrapolating.
            Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # References
        - [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        - [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
        - [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
    '''
    def __init__(self, stack_sizes, R_stack_sizes, num_copies,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time = None,
                 dim_ordering=K.image_dim_ordering(), **kwargs):
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        self.num_copies = num_copies
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max
        self.error_activation = activations.get(error_activation)
        self.A_activation = activations.get(A_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.channel_axis = -3 if dim_ordering == 'th' else -1
        self.row_axis = -2 if dim_ordering == 'th' else -3
        self.column_axis = -1 if dim_ordering == 'th' else -2

        # shape: C x F/2
        self.permutations = []
        for i_stack,dim in enumerate(R_stack_sizes):
            permutations = []
            indices = np.arange(dim / 2)            
            for i in range(self.num_copies):
                np.random.shuffle(indices)
                permutations.append(np.concatenate(
                    [indices,
                     [ind + dim / 2 for ind in indices]]))
                # C x F (np)
            self.permutations.append(np.vstack(permutations))
        
        super(PredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def get_alstm_dim(self, name):
        if name == 'inputs':
            return dim * 4.5
        if name in ['states', 'cells']:
            return dim
        if name == 'mask':
            return 0
        return super(PredNet, self).get_dim(name)    
        
    def get_output_shape_for(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers,)
        else:
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2**self.output_layer_num
            out_nb_col = input_shape[self.column_axis] / 2**self.output_layer_num
            if self.dim_ordering == 'th':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    # The activation function that bound values between 0 and 1
    # input_: B x F
    def bound(self, input_):                
        sq = input_ ** 2        
        _,_,_,dim = input_.get_shape().as_list()
        d = K.sqrt(K.maximum(
            1.0, sq[:,:,:,:dim / 2] + sq[:,:,:,dim / 2:]))
        d = K.concatenate([d, d], axis=-1)
        return input_ / d

    # input: B x F
    # output: C x B x F
    def permute(self, input, dim):
        inputs_permuted = []
        permutation = self.permutations[dim]        
        for i in range(permutation.shape[0]):            
            inputs_permuted.append(
                K.expand_dims(
                    tf.transpose(
                        tf.gather(
                            tf.transpose(input,[3,0,1,2]),
                        permutation[i]),
                    [1,2,3,0]),0))            
        return K.concatenate(inputs_permuted, axis=0)
   
    def get_initial_states(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = K.zeros_like(x)  # (samples, timesteps) + image_shape
        non_channel_axis = -1 if self.dim_ordering == 'th' else -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)  # (samples, nb_channels)

        initial_states = []
        states_to_pass = ['r', 'c', 'e'] # 'r': representation layer; 'c' ?; 'e': error / tuning layer
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}
        if self.extrap_start_time is not None:
           states_to_pass.append('ahat')  # pass prediction in states so can use as actual for t+1 when extrapolating
           nlayers_to_pass['ahat'] = 1
        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u == 'r':
                    stack_size = self.R_stack_sizes[l]
                elif u == 'c':
                    stack_size = self.R_stack_sizes[l]*self.num_copies
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                output_size = stack_size * nb_row * nb_col  # flattened size

                reducer = K.zeros((input_shape[self.channel_axis], output_size)) # (nb_channels, output_size)
                initial_state = K.dot(base_initial_state, reducer) # (samples, output_size)
                if self.dim_ordering == 'th':
                    if u == 'c':
                        output_shp = (self.num_copies,-1, self.R_stack_sizes[l], nb_row, nb_col)
                    else:
                        output_shp = (-1, stack_size, nb_row, nb_col)                    
                else:
                    if u == 'c':
                        output_shp = (self.num_copies,-1,nb_row, nb_col, self.R_stack_sizes[l])
                    else:
                        output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        if K._BACKEND == 'theano':
            from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
            initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0, int)]  # the last state will correspond to the current timestep
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'ik', 'ok', 'o', 'a', 'ahat']} #i,f,c,ik,ok,o 6 elements of Associative LSTM stands for input, forget, memory, key, and output gate

        for l in range(self.nb_layers):
            for c in ['i', 'f', 'o']:
                act = self.LSTM_inner_activation
                self.conv_layers[c].append(Convolution2D(self.R_stack_sizes[l]/2, self.R_filt_sizes[l], self.R_filt_sizes[l], border_mode='same', activation=act, dim_ordering=self.dim_ordering))

            for c in ['ik','ok','c']:
                act = self.LSTM_activation if c == 'c' else self.bound # need to define bound operation as an act like LSTM_activation
                self.conv_layers[c].append(Convolution2D(self.R_stack_sizes[l], self.R_filt_sizes[l], self.R_filt_sizes[l], border_mode='same', activation=act, dim_ordering=self.dim_ordering))
                
            act = 'relu' if l == 0 else self.A_activation
            self.conv_layers['ahat'].append(Convolution2D(self.stack_sizes[l], self.Ahat_filt_sizes[l], self.Ahat_filt_sizes[l], border_mode='same', activation=act, dim_ordering=self.dim_ordering))

            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(Convolution2D(self.stack_sizes[l+1], self.A_filt_sizes[l], self.A_filt_sizes[l], border_mode='same', activation=self.A_activation, dim_ordering=self.dim_ordering))

        self.upsample = UpSampling2D(dim_ordering=self.dim_ordering)
        self.pool = MaxPooling2D(dim_ordering=self.dim_ordering)

        self.trainable_weights = []
        nb_row, nb_col = (input_shape[-2], input_shape[-1]) if self.dim_ordering == 'th' else (input_shape[-3], input_shape[-2])
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c == 'a':
                    nb_channels = 2 * self.R_stack_sizes[l]
                else:
                    nb_channels = self.stack_sizes[l] * 2 + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        nb_channels += self.R_stack_sizes[l+1]
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
                if self.dim_ordering == 'tf': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.states = [None] * self.nb_layers*3

        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(self.extrap_start_time, int)

    def step(self, a, states):
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            a = K.switch(t >= self.t_extrap, states[-2], a)  # if past self.extrap_start_time, the previous prediction will be treated as the actual

        c = []
        r = []

        e = []

        # top-down propagation
        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:
                inputs.append(r_up)

            inputs = K.concatenate(inputs, axis=self.channel_axis)
            i = self.conv_layers['i'][l].call(inputs)
            i = K.concatenate([i,i],axis=-1)
            f = self.conv_layers['f'][l].call(inputs)
            f = K.concatenate([f,f],axis=-1)
            o = self.conv_layers['o'][l].call(inputs)
            o = K.concatenate([o,o],axis=-1)
            ik = self.bound(self.conv_layers['ik'][l].call(inputs)) # input key
            ik = self.permute(ik,l)
            ok = self.bound(self.conv_layers['ok'][l].call(inputs)) # output key
            ok = self.permute(ok,l)            
            u = self.bound(self.conv_layers['c'][l].call(inputs)) # update

            # 1 x B x nR x nC x F , C x B x nR x nC x F --> C x B x nR x nC x F
            f_x_c = K.expand_dims(f,0) * c_tm1[l];
            # B x nR x nC x F , B x nR x nC x F --> 1 x B x nR x nC x F
            i_x_u = K.expand_dims(i*u,0)            
            _c = f_x_c  + complex_mult(ik,i_x_u)

            # C x B x nR x nC x F, C x B x nR x nC x F
            o_x_c = complex_mult(ok,_c)            
            _r = o * self.bound(K.mean(o_x_c, axis=0))            
            
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        # bottom-up propagation
        for l in range(self.nb_layers):
            ahat = self.conv_layers['ahat'][l].call(r[l]) # Add Warp Unit here
            if l == 0:
                ahat = K.minimum(ahat, self.pixel_max) # SatLU
                frame_prediction = ahat

            # compute errors
            e_up = self.error_activation(ahat - a)
            e_down = self.error_activation(a - ahat)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))

            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e
        if self.extrap_start_time is not None:
            states += [frame_prediction, t + 1]
        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'A_activation': self.A_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        base_config = super(PredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
