import numpy as np
import tensorflow as tf
import sys
import sympy as sp
from sympy.utilities.lambdify import lambdify
from utils_dsr import functions, pretty_print
from utils_dsr.symbolic_network import SymbolicNetL0
from utils_dsr.regularization import l12_smooth


funcs = functions.default_func
print(funcs)
x_dim = 3
# Random data for a simple function
x = np.random.rand(100, x_dim) * 2 - 1
#y = x**2
y = np.expand_dims((x[:,0]**2)+(x[:,1]**2)+(x[:,2]**2),axis=1)
#print(y)
print(x.shape, y.shape)

# Set up TensorFlow graph for the EQL network
tf.compat.v1.disable_eager_execution()
x_placeholder = tf.compat.v1.placeholder(shape=(None, x_dim), dtype=tf.float32)
print(x_placeholder.shape, tf.shape(x_placeholder))
sym = SymbolicNetL0(symbolic_depth=2, funcs=funcs, init_stddev=0.5)
y_hat = sym(x_placeholder)

# Set up loss function with L0.5 loss
mse = tf.compat.v1.losses.mean_squared_error(labels=y, predictions=y_hat)
loss = mse + 1e-2 * sym.get_loss()
# Set up TensorFlow graph for training
opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-2)
train = opt.minimize(loss)

# Training
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(2000):
        sess.run(train, feed_dict={x_placeholder: x})
    
    # Print out the expression
    weights = sess.run(sym.get_weights())
    if any([np.any(np.isnan(i)) for i in weights]):
        weights = [np.nan_to_num(i, nan = 0.0) for i in weights]

    expr = pretty_print.network(weights, funcs, ['x', 'y', 'z'])
    str_expr = str(expr)
    while 'x' not in str_expr or 'y' not in str_expr or 'z' not in str_expr:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(train, feed_dict={x_placeholder: x})
        weights = sess.run(sym.get_weights())
        if any([np.any(np.isnan(i)) for i in weights]):
            weights = [np.nan_to_num(i, nan = 0.0) for i in weights]

        expr = pretty_print.network(weights, funcs, ['x', 'y', 'z'])
        str_expr = str(expr)
    
    x_sym = sp.symbols('x')
    y_sym = sp.symbols('y')
    z_sym = sp.symbols('z')

    expr = sp.sympify(str_expr)
    model_selection = lambdify((x_sym, y_sym, z_sym), expr)
    y_hat = model_selection(x[:,0], x[:,1], x[:,2])
    
    loss = np.sum((y_hat.flatten()-y.flatten())**2)
    print(expr, loss, sep="\n")
