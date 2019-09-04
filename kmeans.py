import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IRIS.csv')
x = df.iloc[:, [0, 1, 2, 3]].values

def costKmeans(x, c, centroid, K):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        cost += np.square(np.linalg.norm(centroid[c[i]] - x[i]))
        
    return cost/m

def Kmeans(X, Kval):
    
    initialisation_count = 50
    m = X.shape[0]
    n = X.shape[1]
    niter = 50
    J = -1
    c_final = []
    centroids_final = []
    
    for i in range(initialisation_count):
        
        centroids = []        
        #Random initialisation of centroids
        for i in range(0, Kval):
            centroids.append(X[np.random.randint(X.shape[0])])

        for i in range(niter):
            
            c = []
            #Cluster Assignment
            for j in range(m):

                centroiddist = np.linalg.norm(X[j] - centroids[0])
                minic = 0

                for k in range(1,Kval):
                    temp = np.linalg.norm(X[j] - centroids[k])            
                    if temp < centroiddist :
                        minic = k
                        centroiddist = temp
                
                c.append(minic)

            #Centroid Movement
            temp = np.zeros(shape = (Kval, n))
            count = np.array([0 for i in range(Kval)])
            for i in range(m):
                temp[c[i]] += X[i]
                count[c[i]]+= 1

            for i in range(Kval):
                if count[i] == 0:
                    continue
                centroids[i] = temp[i]/count[i]
            
        J_init = costKmeans(X, c, centroids, Kval)
            
        if J_init < J or J == -1 :
            J = J_init
            c_final = c
            centroids_final = centroids
            
        
    return (J,c)



#J = []
#for i in range(2,8):
#    (j,c) = Kmeans(x,i)
#    J.append(j)
    
#plt.plot(range(2,8), J)

#(cost, clusters) = Kmeans(x,3)
