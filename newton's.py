# this implementation is written with tensorflow
# newton's method for multivariate function

import numpy as np
import tensorflow as tf

def cons(x):
	return tf.constant(x, dtype=tf.float32)

def compute_hessian(fn, vars):
	for v1 in vars:
		temp=[]
		for v2 in vars:
			temp.append(tf.gradients(tf.gradients(fn, v2)[0], v1)[0])
		for t in temp:
			if t==None:
				temp=[cons(0)]
			else:
				temp=t
			temp=tf.pack(temp)
			mat.append(temp)
	mat=tf.pack(mat)
	return mat



def compute_grads(fn, vars):
	grads=[]
	for v in vars:
		grads.append(tf.gradients(fn, v)[0])
	return tf.reshape(tf.pack(grads), shape=[2, -1])


def optimize(all_variables, update):
	optimize_variables=[]
	for i in range(len(all_variables)):
		optimize_variables.append(all_variables[i].assign(all_variables[i]-alpha*tf.squeeze(update[i])))
	return tf.pack(optimize_variables)



def main():
	x=tf.Variable(np.random.random_sample(), dtype=tf.float32)
	y=tf.Variable(np.random.random_sample(), dtype=tf.float32)

	alpha=cons(0.1)
	f=cons(0.5)*tf.pow(x, 2)+cons(2.5)*tf.pow(y, 2)
	all_variables=[x, y]

	hessian=compute_hessian(f, all_variables)
	inverse=tf.matrix_inverse(hessian)
	gradients=compute_grads(f, all_variables)
	update=tf.unpack(tf.matmul(hessian_inv, gradients))

	optimize_op=optimize(all_variables, updates)

	session=tf.Session()
	session.run(tf.initialize_all_variables())

	func=np.inf
	for i in range(100):
		prev=func
		v1, v2, func=session.run([x, y, f])
		print(v1, v2, func)
		session.run(optimize_op)

if __name__==__main__():
	main()