from numpy import *
import pandas as pd

# assume we are given the example of y=mx+b
# m:= slope, b:= y-intercept
def compute_error_for_line_given_points(b, m, points):
	total_error=0
	for i in range(0, len(points)):
		x=points[i, 0]
		y=points[i, 1]
		total_error+=(y-(m*x+b))**2
	return total_error/float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
	# initialize the gradient
	b_gradient=0
	m_gradient=0
	N=float(len(points))
	for i in range(0, len(points)):
		x=points[i,0]
		y=points[i,1]
		b_gradient+=-(2/N)*(y-((m_current*s)+b_current))
		m_gradient+=-(2/N)*x*(y-((m_current*x)+b_current))
	updated_b=b_current-learning_rate*b_gradient
	updated_m=m_current-learning_rate*m_gradient
	return [updated_b, updated_m]

def gradient_descent_runner(points, start_b, start_m, learning_rate, num_iterations):
	b=start_b
	m=start_m
	for i in range(num_iterations):
		b, m=step_gradient(b, m, np.array(points), learning_rate)
	return [b,m]


def run():
	points=pd.read_csv('data.csv')
	learning_rate=0.001
	initial_b=0
	initial_m=0
	num_iterations=1000
	[b, m]=gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

if __name__=='__main__':
	run()

