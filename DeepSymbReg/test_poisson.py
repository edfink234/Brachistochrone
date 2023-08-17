import numpy as np
import tensorflow as tf
import sys
import sympy as sp
from sympy.vector import CoordSys3D, laplacian
from sympy.utilities.lambdify import lambdify
from utils_dsr import functions, pretty_print
from utils_dsr.symbolic_network import SymbolicNetL0
from utils_dsr.regularization import l12_smooth


funcs = functions.default_func
x_dim = 1
# Random data for a simple function
x = np.random.rand(100, x_dim)
U = np.random.rand(100, x_dim)**4
C = CoordSys3D('C')
U_sym = C.x**4
print("Original function =",U_sym)
lap_U_sym = laplacian(U_sym)
print(f"Output function = {lap_U_sym} = Laplacian(Original function)")
y_func = lambdify(C.x, sp.sympify(lap_U_sym))
y = y_func(U)

# Set up TensorFlow graph for the EQL network
tf.compat.v1.disable_eager_execution()
x_placeholder = tf.compat.v1.placeholder(shape=(None, x_dim), dtype=tf.float32)
sym = SymbolicNetL0(symbolic_depth=2, funcs=funcs, init_stddev=0.5)
y_hat = sym(x_placeholder)
#print(y_hat, y)
y = np.full((100, 1), fill_value = y)
print("y.shape, y_hat.shape", y.shape, y_hat.shape)

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

    expr = pretty_print.network(weights, funcs, ['x'])
    str_expr = str(expr)
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x_placeholder: x})
        weights = sess.run(sym.get_weights())
        if any([np.any(np.isnan(i)) for i in weights]):
            weights = [np.nan_to_num(i, nan = 0.0) for i in weights]

        expr = pretty_print.network(weights, funcs, ['x'])
        str_expr = str(expr)
    
    x_sym = sp.symbols('x')

    expr = sp.sympify(str_expr)
    model_selection = lambdify((x_sym), expr)
    y_hat = model_selection(x)
    
    loss = np.sum((y_hat.flatten()-y.flatten())**2)
    print(expr, loss, sep="\n")
