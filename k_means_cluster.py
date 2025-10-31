"""
A simple K-Means implementation

Methodology for the K Means algorithm:

    Choose value for K
    Randomly select K featuresets to start as your centroids
    Calculate distance of all other featuresets to centroids
    Classify other featuresets as same as closest centroid
    Take mean of each class (mean of all featuresets by class), making that mean the new centroid
    Repeat steps 3-5 until optimized (centroids no longer moving)
"""
#defaultdict = Dictionary, das automatisch leere Listen erzeugt (praktisch für Cluster-Listen)
#typing = nur für saubere Typ-Hints im Code
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

'''
matplotlib: zum Plotten der Cluster
numpy: Mathe & Arrays
cityblock: Manhattan-Distanz‐Funktion
SAMPLE_DATA: Beispiel-Dataset aus Repo
'''
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cityblock
from make_clusters import SAMPLE_DATA

#Alias für Datentypen, die als Input erlaubt sind:
ClusterData = Union[List, np.array, np.ndarray, DataFrame]

#Klasse, die k-means implementiert:
class KMeansCluster:
    """Simple K-Means Clustering Model"""
    
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        #Sicherstellen, dass Methode gültig ist:
        if method not in {'euclidean', 'manhattan'}:
            raise ValueError('Method must be one of "euclidean" or "manhattan"')
        #Speichert Parameter in Objekt:
        self.k = k                                                                      #k = Anzahl Cluster
        self.tol = tol                                                                  #tol: Toleranz für Konvergenz (wie stark dürfen Centroids sich noch bewegen)
        self.max_iter = max_iter                                                        #max_iter: Abbruch nach X Iterationen
        self.method = method                                                            #method: Distanzmethode (euclidean oder manhattan)

        self.centroids: Dict[int, Any] = {}                                             #centroids: speichert die aktuellen Cluster-Zentren {cluster_id: point}
        self.clusters: Dict[int, Any] = defaultdict(list)                               #clusters: speichert zugewiesene Punkte {cluster_id: [points]} und defaultdict(list) = automatisch leere Liste
        self.cluster_distances: Dict[int, Any] = {}                                     #cluster_distances: wird später für Cluster-Qualität genutzt

    #Zufällige Start-Zentren
    def initialize_centroids(self, data: ClusterData) -> "KMeansCluster":
        """Get initial centroids"""
        choices = np.random.choice([*range(len(data))], size=self.k, replace=False)     #np.random.choice → wählt zufällig k Indexe
        for i in range(self.k):
            self.centroids[i] = data[choices[i]]
        return self

    @staticmethod
    def _convert_to_array(data: ClusterData) -> np.array:
        """Function to convert data into ndarray"""                                     #Konvertiert Input in NumPy-Array, egal ob Liste/DF
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (List, Tuple)):
            return np.array(data)
        if isinstance(data, DataFrame):
            return DataFrame.values
        raise TypeError

    #Distanzberechnung:
    def get_distance(self, arr_1: ClusterData, arr_2: ClusterData) -> float:
        """Calculate squared distance between two points"""
        arr_1 = self._convert_to_array(arr_1)
        arr_2 = self._convert_to_array(arr_2)
        #Konvertiert zu Arrays:
        if self.method == 'manhattan':
            dist = cityblock(arr_1, arr_2)
        else:
            dist = np.linalg.norm(arr_1 - arr_2)
        return dist

    @staticmethod
    def find_min_idx(distances: List) -> int:
        """Find the index of the minimum distance"""
        return distances.index(min(distances))                                          #Gibt Index des nächstgelegenen Zentroids zurück

    #Zuweisung: Welcher Punkt in welchen Cluster?
    def assign_cluster(self, data: ClusterData) -> "KMeansCluster":
        """Function to assign a point to a centroid"""
        #Reset aller Cluster:
        data = self._convert_to_array(data)
        self.clusters.clear()
        #für jeden Punkt → Distanz zu jedem Centroid berechnen
        for point in data:
            distance = []
            for centroid in self.centroids.values():
                distance.append(self.get_distance(point, centroid))
            #finde kleinste Distanz → Punkt zu diesem Cluster
            cluster = self.find_min_idx(distance)
            self.clusters[cluster].append(point)
        return self

    #Centroid Update
    def update_centroid(self) -> "KMeansCluster":
        """Function to update centroids"""
        for cluster in self.clusters:
            new_centroid = np.mean(np.array(self.clusters[cluster]), axis=0)            #Mittelwert aller Punkte im Cluster = neuer Zentroid  ==>Standard-K-Means Kernidee!
            self.centroids[cluster] = new_centroid
        return self

    #Prüfen, ob Centroid-Bewegung klein genug:
    def meets_tolerance(self, old_centroids: ClusterData, new_centroids: ClusterData) -> bool:
        """Function to detect convergence"""
        return np.abs(self.get_distance(old_centroids, new_centroids)) <= self.tol

    #Trainingsloop:
    def fit(self, data: ClusterData, verbose=0):
        """
        Function to fit K-Means object to dataset.

        :param data: Dataset of type [list, DataFrame or NDArray]
        :param verbose: Integer [0, 1, 2] indicating verbosity of printouts
        :return: Dictionary of clusters and points
        """
        if verbose not in {0, 1, 2}:
            raise ValueError('Verbose must be set to {0, 1, 2}')
        data = self._convert_to_array(data)
        self.initialize_centroids(data).assign_cluster(data)                           #Start-Zentren und Erste Zuordnung
        if verbose > 0:
            print(f'Initial clusters: {self.clusters.items()}')
        converged = {}
        i = 0
        #jede Iteration neu zuordnen:
        while i <= self.max_iter:
            self.assign_cluster(data)
            #Wichtig: Für jede Cluster-Mitte prüfen: bewegt sie sich noch?
            for centroid in self.centroids:
                old_centroid = self.centroids[centroid]
                new_centroid = self.update_centroid().centroids[centroid]
                converged[centroid] = self.meets_tolerance(old_centroid, new_centroid)
            if verbose > 1:
                print(f'Iteration number: {i+1}')
            #Wenn alle Centroids stabil → fertig.
            if all(converged.values()):
                break
            i += 1
        if verbose > 0:
            print(f'Final clusters: {self.clusters.items()}')
            print(f'Converged in {i+1} iterations.')
        return self

    #Plot Evaluation - optional; Nicht algorithmisch wichtig, nur fürs Anzeigen.
    def plot(self) -> None:
        """Plot clusters and centroids"""
        for cluster in self.clusters.values():
            plt.scatter(np.array(cluster)[:, 0], np.array(cluster)[:, 1], alpha=0.5)
        for point in self.centroids.values():
            plt.scatter(point[0], point[1], marker='x', s=100, c='red')
        plt.show()

    def get_cluster_ssd(self) -> Dict[int, Any]:
        """Get within cluster sums of squared distances"""
        cluster_distances: Dict[int, Any] = {}
        for cluster in self.centroids:
            centroid_dist = 0
            for point in self.clusters[cluster]:
                centroid_dist += self.get_distance(self.centroids[cluster], point)
            cluster_distances[cluster] = centroid_dist
        return cluster_distances

    def eval_metrics(self) -> Dict[str, Union[Dict[int, float], Dict[str, float]]]:
        """Get cluster evaluation metrics"""
        clusters: Dict[int, float] = self.get_cluster_ssd()
        total_ssd: float = sum(clusters.values())
        return {'cluster_dist': clusters, 'total_dist': total_ssd}


def main():
    """Main function"""
    kmeans = KMeansCluster(k=3)
    kmeans.fit(SAMPLE_DATA, verbose=2)
    print(list(kmeans.clusters.values()))
    kmeans.plot()
    kmeans.eval_metrics()


if __name__ == '__main__':
    main()
