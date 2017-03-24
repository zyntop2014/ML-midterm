##############################################
# COMS 4771 Machine Learning Midterm
# Name: Yanan Zhang
# UNI: yz3054
##############################################

# import all necessary modules here
# see midterm instruction for requirements
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pickle


def predict(X_test):
    """This function takes a dataset and predict the class label for each data point
	should be in.

	Parameters
	----------
	dataset: M X D numpy array
		A dataset represented by numpy-array

	Returns
	-------
	M x 1 numpy array
		Returns a numpy array of predicted class labels
    """
    # Your code here
     
    model = pickle.load(open("model.pkl", "rb" ))
    M= X_test.shape[0]
    D= X_test.shape[1]
    
    
        #original data
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    pca=PCA(n_components=3)
    X_reduced = pca.fit_transform(X_test)
    y_pred=model.predict(X_reduced) 
    
    print (X_reduced)     
            
    df = pd.DataFrame(np.vstack([X_reduced]))
    df["label"]= np.hstack([y_pred])
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_pred,
                       cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions for testing data")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.savefig('rawdataPCAplot.png') 
    plt.show()
        
    return (y_pred)
   
    
    
    
if __name__ == "__main__":
    X_test = pickle.load( open( "test1.pkl", "rb" ) )
    y = pickle.load( open( "Data_y.pkl", "rb" ) ) 
   
    y_pred=predict(X_test)
   
    n=0
    total=y.shape[0]
    #scores for fitting different data set
    for i, j in zip(y_pred, y):
        if i==j:
            n=n+1
            
    test_accuracy = float(n)/total
   
    
    print ("\n============")
    print ("Tesing data...")
    print("Testing accuracy: {}".format(test_accuracy))

     
     
        