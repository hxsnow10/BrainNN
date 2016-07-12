'''
Lazar, Andreea, Gordon Pipa, Jochen Triesch, Andreea Lazar, Gordon Pipa, and Jochen Triesch. “SORN: A Self-Organizing Recurrent Neural Network.” Frontiers in Computational Neuroscience 3 (2009): 23. doi:10.3389/neuro.10.023.2009.

implemented in theano_blocks.

i make some chage to the original model.

TODO:
这里有些问题，应该怎么update。
是batch还是一个一个？
是一个inputs完了update还是每个time_step都update。

从生物学上来说，自然是一个一个每个time_step来update。
如果我们期望batch的话，那最好等所有跑完再update，可以把值记录下来
如果要batch updates并且是每个timestep的话，那需要为每个sample都备份一个
weights,最后再把值综合到一起。

前者更像一个一般的智能系统，后者则泛乎到了批处理智能。显然人的批处理很弱。
我把2个都实现一下。
'''

class SORN(SimpleRecurrent):
    '''
    similar to simple_rnn except:
    1) some difference about states updates
        1.1 
    2) updates by local learning rule every time step
    '''
    
    @lazy(allocation=['dim'])
    def __init__(self, dim, inhi_dim, activation, **kwargs):
        self.dim = dim
        self.inhi_dim = inhi_dim

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
        self.WEE=shared_floatx_nans((self.dim, self.dim),
                                                  name="WEE")
        self.WEI=shared_floatx_nans((self.inhi_dim, self.dim),
                                                  name="WEI")
        self.WIE=shared_floatx_nans((self.dim, self.inhi_dim),
                                                  name="WIE")
        self.TE=shared_floatx_nans((self.dim,),name="TE")
        self.TI=shared_floatx_nans((self.inhi_dim,),name="TI"))
        self.initial_state_E=shared_floatx_zeros((self.dim,),
                                                   name="initial_state_E")
        self.initial_state_I=shared_floatx_zeros((self.inhi_dim,),
                                                   name="initial_state_I")
        
        add_role(self.WEE,WEIGHT)
        add_role(self.WEI,WEIGHT)
        add_role(self.WIE,WEIGHT)
        add_role(self.TE,WEIGHT)
        add_role(self.TI,WEIGHT)
        add_role(self.initial_state_E, INITIAL_STATE)
        add_role(self.initial_state_I, INITIAL_STATE)

        self.parameters = [ self.WEE, self.WEI, self.WIE, self.TE, self.TI,
                self.initial_state_E, self.initial_state_I]


    def _initialize(self):
        for weights in self.parameters[:5]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], 
            states=['states_E', 'states_I', 'weights'], 
            outputs=['states_E', 'states_I', 'weights'], contexts=[])
    def apply(self, inputs, states_E, states_I, weights, mask=None, updates=True):
        """Apply the simple transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        WEE,WEI,WIE,TE,TI=weights
        #TODO: change to batch dot
        def batch_dot(S,W):
            result,updates=scan(fn=lambda s,w:tensor.dot(s,w),
                    inputs=[S,W])
            return result

        next_states_E = inputs + batch_dot(states_E, WEE) +
            batch_dot(states_E, WEI)-TE
        next_states_E = self.children[0].apply(next_states_E)

        next_states_I= batch_dot(states_I, WIE)-TI
        next_states_I= self.children[0].apply(next_states_I)

        if mask:
            next_states_E = (mask[:, None] * next_states_E +
                           (1 - mask[:, None]) * states_E)
            next_states_I = (mask[:, None] * next_states_I +
                           (1 - mask[:, None]) * states_I)
        
        next_weights=weights
        if updates:
            def hebbian(x1,x2):
                z=x1[:,None]*x2[None,:]
                return z-z.transe
            WEE=WEE+scan(fn=hebbian, inputs=[states_E, next_states_E])[0]
            WEE=(WW)/sum(WEE,0)
            TE=TE+t*(next_states_E-Hp)

            next_weights=[WEE,WEI,WIE,TE,TI]
        return next_states_E, next_states_I, next_weights
    
    @recurrent(sequences=['inputs', 'mask'], 
            states=['states_E', 'states_I', 'updated'], 
            outputs=['states_E', 'states_I', 'updated'], contexts=[])
    
    def apply2(self, inputs, states_E, states_I, updated, mask=None, updates=True):
        """Apply the simple transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        WEE,WEI,WIE,TE,TI=weights
        next_states_E = inputs + tensor.dot(states_E, WEE) +
            tensor.dot(states_E, WEI)-TE
        next_states_E = self.children[0].apply(next_states_E)

        next_states_I= tesnor.dot(states_I, WIE)-TI
        next_states_I= self.children[0].apply(next_states_I)

        if mask:
            next_states_E = (mask[:, None] * next_states_E +
                           (1 - mask[:, None]) * states_E)
            next_states_I = (mask[:, None] * next_states_I +
                           (1 - mask[:, None]) * states_I)
        
        if updates:
            WEE=WEE+s(next_states_E*state_E
            WEE=
            TE=
            updated=[]
        return next_states_E, next_states_I, updated

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.initial_states_E[None, :], batch_size, 0),
            tensor.repeat(self.initial_states_I[None, :], batch_size, 0),
            [tesnor.repeat(p) for p in self.parameters[:5]]



