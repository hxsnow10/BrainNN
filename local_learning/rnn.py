'''
一般来说，这个模型应该用RNN解决。inputs+states--->（new_states,output)
多重时间尺度等价于权值大小
'''
class simple_nn():
    '''不用区分inputs，states与outputs.
    inputs即与图无依赖的states
    outputs即不能提供依赖的states
    整个作为一个rnn即states(不包括inputs)的update过程
    '''

    @recurrent(sequences=[''],outputs=['output'])
    def apply(self, inputs, states, mask=None):
        L1,L2=states
        I1,I2=inputs
        newL1=self.alpha*L1+(1-self.alpha)*\
    tensor.relu((tensor.dot(inputs,self.W)+tensor.dot(L2,self.W))),\
        newl2=
        output=
        return newL1,newL2,output

inputs_series=T.matrix()# in fact, time se
snn=simple_nn()
output_series=snn.apply(inputs_seriee)

'''
leaky layer
'''


'''
如果说我们想把一个原本的计算图自动转化为这样的RNN
我们定义一个函数  transform(graph, states)
由于一般的层次没有状态性，我们可以使用RNN一层叠加来组成一个有状态的前馈网络
'''
def transform (gh, states):
    '''
    map gh to states_updates
    return to what similar to what snn.apply really return
    '''
    new_states=


mlp = MLP([Tanh(),Tanh(), Identity()], [1, 100, 100, 1], 
          weights_init=IsotropicGaussian(0.01),
          biases_init=Constant(0), seed=1)
rnn = SimpleRecurrent(100,activation=)

mlp.initialize()
x = tensor.vector('numbers')
y = tensor.vector('roots')
y_hat=rnn.apply(mlp.apply(x[:,None]), iterate=False)
cg=graph(y_hat,inputs=[],outputs=[],states=[x,y_hat])

'''
learning
'''

def get_gradients():
    '''
    local learning 仅仅跟一行微分方程的变量相关
    '''

gradients=get_gradients()

'''
combine all together
'''

main_loop = MainLoop(
    GradientDescent(
        gradients=gradients, parameters=ComputationGraph(cost).parameters,
        step_rule=Scale(learning_rate=0.001)),
    get_data_stream(range(100)),
    model=Model(cost),
    extensions=[
        Timing(),
        FinishAfter(after_n_batches=num_batches),
        DataStreamMonitoring(
            [cost], get_data_stream(range(100, 200)),
            prefix="test"),
        TrainingDataMonitoring([cost], after_epoch=True),
        Checkpoint(save_to),
        Printing()])

'''一个细节，就是当inputs未倒位，那需要这个模型能处理不定长的输入，补0
浪费计算，或者说我们让0的时候不浪费计算。
‘’‘

