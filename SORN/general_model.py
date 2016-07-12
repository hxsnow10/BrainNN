'''一个general的RNN
应该定义的计算结构
'''


class RNN(SimpleRecurrent):
    '''
    some abstract model to define structural RNN.
    abstracts of RNN is dynamic systems, all variables has it's dynamics, \
    when in NN some as states some as weights.

    when NN become more complex, for example biology models, LSTM, multi\
    scale weights, simple RNN return complex dynamics, although we can 
    say neuron and weights becom more complex.   

    here we follow some iteration uodates form. given compoments(states), 
      states updates (network), learning rule(weights update rule)

    
    Parameters
    ------------

    Example
    -----------
    inputs=tensor.vector()
    en_states=
    in_states=
    out=
    SORN=RNN([],
    '''
    
    @lazy(allocation=['dim'])
    def __init__(self, compoments,compoments_type,weights, states_update,weight   **kwargs):
        self.compoments = compoments
        self.states=states#TODO
        self.weights=weights
        self.states_update=states_update
        self.weights_update=weights_update

        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(SimpleRecurrent, self).__init__(**kwargs)


    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (SimpleRecurrent.apply.sequences +
                    SimpleRecurrent.apply.states):
            return self.dim
        return super(SimpleRecurrent, self).get_dim(name)

    def _allocate(self):


    def _initialize(self):
        for weights in self.parameters[:5]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], 
            states=['states', 'weights'], 
            outputs=['output'], contexts=[])
    def apply(self, inputs, states, weights, mask=None, updates=True):
        """Apply the simple transition.

        """
        next_states = self.states_updates(states, weights)
        next_weights = self.weights_updates(states, weights, next_states)
        return next_states, next_weights
    
    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.initial_states_E[None, :], batch_size, 0),
            tensor.repeat(self.initial_states_I[None, :], batch_size, 0),
            [tesnor.repeat(p) for p in self.parameters[:5]]
