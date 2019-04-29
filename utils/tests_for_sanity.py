import numpy as np	

def ensure_classification4_prediction_correct():
	

	n = 4
	for i in range(10000):
		x = (np.random.rand(5)*360).astype(np.int32)
		y = (x/int(360/n)).astype(np.int32)
		for x,y in zip(x,y):
			z = 0 if x<90 else 1 if x<180 else 2 if x<270 else 3
			try:
				assert(y == z)
			except:
				print(x,z)