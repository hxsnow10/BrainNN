import theano
import theano.tensor as T
from blocks import UpdatesAlgorithm
from collections import OrderedDict

'''
Here we implement Local Learning, which means learning is about and only 
directly about local virables.

Here we don't emulate biology abolute local, but update with batch.
BP may be implemented with a local variable from BackProgation, we should 
make it $z_t=\alpha z_{t-1}+(1-\aplha) new

很重要的一点：如果local learning，物理层或者底层实现都要做相应的改变

theano， BP的cost上实现为grad&updates，
Local_learning， 由updates直接实现
Feed(inputs), Conv(inputs), Rec(inputs), 


Y=act(X.dot(W)+b)
delt=f(X,Y,W)
hebbian_updates={W:W+delt}

theano实现很简单
blocks声称关于theano透明，实际上main_loop还是封装了的，brick确实可以是透明，
但是所有的brick比较重，继承很多，很难读
langrane比较轻一些。
'''
'''
local learning就是要获取updates。
我们需要为各种theano.op或者blocks.layer定义updates
'''
gradients=local_updates(parameters, outputs, local_update_rule)

def local_updates(parameters, outputs, local_update_rule):
    mul_list=getall(params,outputs)#(param, input, output)
    updates=OrderedDict({})
    for param,input,output,op in mul_list:
        updates[param]=local_update_rule.compute(param,input,output,op) 
    return updates

def get_local_variable(params,outputs):
    var=outputs
    rval=OrderedDict({})
    for param in rval:
        rval[param]=None
    checklist=outputs # a list
    checked=[]
    def is_after_act(y):
        op=y.owner.op
        return type(op)==theano.tensor.elemwise.Elemwise and
            op.scalar_op in acts
    
    # build a tree for output
    while hasattr(checklist[0],'owner'):
        owner=checklist[0].owner
        inputs=owner.inputs
        output=owner.outputs
        op=owner.op
        if is_after_act(output):
            o=output
            search_list,k=inputs,0
            while k<len(search_list):
                s=search_list[k]
                if is_after_act(s):pass
                elif is_dot(s):
                    inputs=s.owner.inputs
                    get (Oj, w, output) to rval
                else:
                    add s.inputs to search_list
    return rval

class local_update_rule():

    def __init__(rule):
        self.compute=rule
        
def hebbian(param, inputs, output, op):
    '''
    + not right above. hebbi an is implemented in neurom. blocks soesn't has neuron concept. hea has MLP,RNN. we must give : output, input, param
    + param use dot to find input, then search a next tanh/..  to find output.
    + In RNN_with time, there may be many with a param.
    + and here param is a matrix, so should update using matrix .
'''
长远的看，需要为每一种算子都定义local learning的规则/形式，
不由有一个问题，是什么约束local learning的形式的？
'''
