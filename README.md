# Using reinforcement learning and symbolic regression to discover the Brachistochrone

Brachistochrone.py just tries to guess faster paths from point A to point B under the influence of only gravity, and updates the fastest path via a matplotlib animation.

RL_Brachistrochrone.py implements a reinforcement learning environment with a neural network that tries to learn the best path under the the influence of only gravity, and also symbolic regression to produce candidate equations based on the coordinates of the best path found by the reinforcement learning agent. The paths are also shown in real-time via a matplotlib animation. An example output is shown in RL_Brachistrochrone.png

## To Install
```
1. cat requirements.txt | xargs -n 1 pip3 install
```

After installing requirements.txt, make the following file modifications (substituting your environtname with `myvenv` and your python version with ```python3.9```:

### myvenv/lib/python3.9/site-packages/rl/agents/ddpg.py 
 - Change `if hasattr(actor.output, '__len__') and len(actor.output) > 1:` to `if hasattr(actor.output, '__shape__') and len(actor.output) > 2:`
 - Change `if hasattr(critic.output, '__len__') and len(critic.output) > 1:` to `if hasattr(critic.output, '__shape__') and len(critic.output) > 2:`
 -  Comment out `if self.uses_learning_phase:`, `critic_inputs += [K.learning_phase()]`, `inputs += [self.training]`

### myvenv/lib/python3.9/site-packages/keras/engine/training_utils_v1.py
 - Comment out the following:

```python
if len(data_shape) != len(shape):
    raise ValueError(
	"Error when checking "
	+ exception_prefix
	+ ": expected "
	+ names[i]
	+ " to have "
	+ str(len(shape))
	+ " dimensions, but got array with shape "
	+ str(data_shape)
    )
```

Then you can run the program:

```
python3 RL_Brachistochrone.py
```
