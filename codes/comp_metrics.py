import sklearn.metrics

# y_GT: ground truth
# y_P: predicted
# this code assumes that the first column (e.g. column 0 = undamaged) is undamaged
# and the second column is damaged (column 1 = damage)
def compute_metrics(y_GT, y_P):
    
    accuracy = sklearn.metrics.accuracy_score(y_GT, y_P > 0.5)
    class_freq = np.sum(y_GT,axis=0)
    accuracy_chance = (class_freq[0]+class_freq[1]-class_freq[1])/(class_freq[1]+class_freq[0])
    
    auc = sklearn.metrics.roc_auc_score(y_GT, y_P)
    MAP = sklearn.metrics.average_precision_score(y_GT, y_P)
    
    return (accuracy, accuracy_chance, auc, MAP)

def print_metrics(y_GT, y_P):
    
    (accuracy, accuracy_chance, auc, MAP) = compute_metrics(y_GT, y_P)
    
    print('accuracy: %4.4f' % accuracy)
    print('accuracy chance: %4.4f' % accuracy_chance)
    print('AUC: %4.4f' % auc)
    print('MAP: %4.4f' % MAP)
    
if __name__ == "__main__":
	
	y_GT = np.array([[0,1],[1,0],[1,0]])
	y_P = np.array([[0,0.5],[1,0],[1,0]])
	
	print_metrics(y_GT, y_P)
	
