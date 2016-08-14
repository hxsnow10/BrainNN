Local Learning
----------

####Concept of local learning
Especially, what is local?
+ Physically and Mathmatically, we can define local as __Spacely near__. Even some __action at a distance__ should in fact work in some 
space where acition take place near. So, local is natural and necessary of action.
+ So local should be natural and necessary for learning, and why we talk about __local learning__? local in Network means states updates
(either dynamic neuron states or weights states or any other) should  only be dependet on network near variables.
+ SGD, whcih in fact implemented using BP locally.If we dont use BP, then update of every of parameter W should dependet of 
(W_in, function betwween W_out and final_out), when the latter is complicated and dependet of a network.This cant be local!
+ I guess all ML algorithm has local version ,when some limit of Network structure and computing order.
+ In the above SGD example, local BP should be limited of the computing order, which mean there should be a center to control.
It is not that local(but maybe such center of course is important).So anther type of local BP is all states/weights update Simultaneously.
This basic ideal is underline all computational neuroscience models.
+ So we define local learning: __all weights update by and only by local variables and time indenpendetly.__
+ As complicated system, states/weights has many levels and cant be distinguished clearly (differ by the rate to change), so we can 
expand the idea to local dynamic system: __all states/weights update by and only local variables and time independetly__ , which is just
expanded huge self-organized Differential Equations.

####Types of local learning
Traditional local learning excludes SGD/BP, i disagree.
Traditional famous local learning includes various hebbian learning, various Neuroplasticity.

####TODO
Above is only my basic understanding of local learning.Thchniqually, we should go deep to 
+ How independet computing order influences traditonal learning algorithm?
+ How to design local learning to support local learning?

