import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(46)
def distance(p1, p2):
    return np.linalg.norm((p1-p2),axis=-1)

def find_centre_of_mass(points):
    return np.mean(points,axis=0)

def centroid_generation(k, x, y):
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    x = np.random.normal(x_mean, x_std,size=(k,1))
    y = np.random.normal(y_mean, y_std,size=(k,1))
    centroids = np.append(x,y,axis=1)
    return centroids

def plot_points(axis,points,point_names=None,color=['b'],marker = 'o',size=40):
    axis.scatter(points[:,0],points[:,1],color=color,marker=marker,s=size)
    if point_names is None:
        return
    for _, ((x,y),t) in enumerate(zip(points,point_names)):
        axis.text(
            x, y,
            f"{t}",
            fontsize=7,
            ha="left",
            va="bottom"
        )
 
def kmeans_clustering(k, x, y, names):
    centroids = centroid_generation(k, x, y)
    centroid_colors = ['r','g','k','y']
    centroid_colors = centroid_colors[:k]
    
    data_points = np.append(x,y, axis=1)
    point_colors = np.array(['b']*x.shape[0])
    
    n=1
    while True:
        for i in range(len(data_points)):
            point = data_points[i]
            distances = distance(centroids,point)
            index = np.argmin(distances)
            print(names[i],centroids,distances)
            point_colors[i] = centroid_colors[index]
        
        new_centroids = np.zeros(centroids.shape)
        for i in range(len(centroids)):
            indices = np.where(point_colors==centroid_colors[i])
            cluster_points = data_points[indices]
            if cluster_points.shape[0] != 0:
                new_centroids[i] = find_centre_of_mass(cluster_points)
        
        if np.allclose(new_centroids, centroids):
            plot_points(plt,centroids,color=centroid_colors,marker='x',size=80)
            plot_points(plt,data_points,names,color=point_colors)
            plt.title(f"Final Clustering")
            plt.show()
            plt.close()
            break
        else: 
            plot_points(plt,centroids,color=centroid_colors,marker='x',size=80)
            plot_points(plt,data_points,names,color=point_colors)
            plt.title(f"Iteration= {n}")
            plt.show()
            plt.close()
            n+=1
            centroids = new_centroids


df = pd.read_excel("UEFAQ_ShootingStats.xlsx")
teams = df["Team"].to_numpy()
x = df[["SoT/90"]]
y = df[["G/SoT"]]
plt.xlabel("SoT/90")
plt.ylabel("G/SoT")

x_norm = (x - np.mean(x))/np.std(x)
y_norm = (y - np.mean(y))/np.std(y)

NUM_CLUSTERS = 3
kmeans_clustering(NUM_CLUSTERS, x_norm, y_norm, teams)
