import numpy

import theano
from theano import tensor

from blocks.algorithms import UpdatesAlogorithm, Scale
from blocks.bricks import MLP, Tanh, Identity
from blocks.bricks.cost import SquaredError
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from fuel.datasets import IterableDataset
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.main_loop import MainLoop

floatX = theano.config.floatX


def get_data_stream(iterable):
    """Returns a 'fuel.Batch' datastream of
    [x~input~numbers, y~targets~roots], with each iteration returning a
    batch of 20 training examples
    """
    numbers = numpy.asarray(iterable, dtype=floatX)
    dataset = IterableDataset( 
        {'numbers': numbers, 'roots': numpy.sqrt(numbers)})
    return Batch(dataset.get_example_stream(), ConstantScheme(90))


def main(save_to, num_batches):
    linear = Linear()
    rnn=SORN()
    x = tensor.vector('numbers')
    states_E, states_I, updates=rnn.apply(linear.apply(x[None, :]))
    y=linear.apply(states_E[-1])
    cost=SquaredError().apply(y[:,None], mlp.apply(states_E[-1]))
    # consider updates about linear from x and to y
    # 1. make all in SORN
    # 2. gradient?
    main_loop = MainLoop(
        UpdatesAlgorithm(
            updates=updates),
        get_data_stream(range(100)),
        model=Model(),
        extensions=[
            Timing(),
            FinishAfter(after_n_batches=num_batches),
            DataStreamMonitoring(
                [cost], get_data_stream(range(100, 200)),
                prefix="test"),
            TrainingDataMonitoring([cost], after_epoch=True),
            Checkpoint(save_to),
            Printing()])
    main_loop.run()
    return main_loop
