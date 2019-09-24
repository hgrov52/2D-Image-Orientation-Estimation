import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# assumed to be numpy
def std(array):	
	array = np.where(array>180,360-array,array)
	return np.std(array)

def plot_confusion_matrix(y_pred, y_true, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	
	title = 'Confusion matrix'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	step = 1 if cm.shape[0]<=10 else cm.shape[0]//10
	ticks = np.arange(0,cm.shape[0],step)#*int(360/min(cm.shape[0],10))
	ax.set(title=title,
			ylabel='True label',
			xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# print quantity in each block, only for few classes bc gets cluttered
	if(len(classes)<10):
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				ax.text(j, i, format(cm[i, j], fmt),
						ha="center", va="center",
						color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax


def test_stats(args, all_pred, all_targ, all_labl, all_diff):
	# ===============================================
	# print basic statistics
	print("For 360 degrees:")
	print("mean:",np.mean(all_diff))
	print("standard deviation:",np.std(all_diff))
	print("median:",np.median(all_diff))

	# ===============================================
	# plot confusion matrix plot
	
	print("max:",max(all_pred))
	classes = np.arange(0,max(all_pred)+1)
	plot_confusion_matrix(all_pred.astype(np.int32), all_targ.astype(np.int32), classes)
	plt.show()

	# ===============================================
	# Plot error histogram
	bins_def = np.arange(max(360,np.max(all_diff)))
	# hist,bins = np.histogram(all_diff,bins = bins_def)
	plt.text(100,50,'mean: {:3f}\nmedian: {:3f}'.format(np.mean(all_diff),np.median(all_diff)))
	plt.hist(all_diff,bins_def)
	plt.title(args.type)
	plt.show()

	# ===============================================
	# distribution of error and label
	

	# ===============================================
	# find worst errors and save them

