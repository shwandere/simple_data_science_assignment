import pandas as pd
from sklearn.cluster import KMeans
import folium
#import cudf

#reading csv
#%cd /content/drive/MyDrive/Colab_first_mount/
data = pd.read_csv("events.csv")
print(data[["latitude"]])
data_series = data.copy()
data_series = data_series[["latitude","longitude"]].squeeze()
#getting lat long only
spatial = data[["latitude","longitude"]]

#clustering on the basis of lat long
#create kmeans model/object
kmeans = KMeans(init="random", n_clusters=16, n_init=10, max_iter=300, random_state=42)
#do clustering
kmeans.fit(spatial)
#save results
labels = kmeans.labels_

#send back into dataframe
data['cluster'] = labels
#display no. of members of each clustering
_clusters = data.groupby('cluster').count()
#print(_clusters)

#finding the cluster where any row count is max I have pcked latitude
#print(max(_clusters['latitude']))
max_cluster_index = _clusters['latitude'].argmax()
#print("max cluster index")
#print(max_cluster_index)
#getting data where cluster is of max size
severe_places = data[data['cluster']==max_cluster_index]
#print(severe_places)
#cluster on the basis of state
_cluster_state = severe_places.groupby('state').count()
#print(_cluster_state)
#finding max cluster
state_max = _cluster_state['latitude'].argmax()
print("state at high severity risk is "+_cluster_state.iloc[state_max].name)
#severe_places[state_max]

###########################################################################

#most affected year
#print(data.iloc[0]['date'][0:4])
data['year'] = data['date'].apply(lambda x: int(x[0:4]))
_year_affected = data.groupby('year').count()
most_affected_year = _year_affected['latitude'].argmax()
#print("most affected year is ")
print(_year_affected.iloc[most_affected_year].name)

###########################################################################

#total no of events occuring at a point in around 50 kms area
#means go to every row,calculate distance from lat long, find areas which are in 50 kms range, count them
from math import cos, asin, sqrt, pi
#from haversine import haversine, Unit
#!pip install pyspark
#!pip install swifter 
#import swifter
# Import necessary libraries 
#from pyspark.sql.functions import col 
#import multiprocessing
#import numpy as np
import pyspark.pandas as ps
#df = ps.DataFrame(data, ["latitutde", "longitude","county"])
df = data[["latitude","longitude","county"]]


def distance(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2

    return 2 * r * asin(sqrt(a))

nearby = []
print(spatial[["latitude"]])


for i in range(len(spatial[["latitude"]])):
    print(i)
    point = spatial.iloc[i]
    #print(spatial['latitude'])
    point_county =  df.iloc[i]["county"]
    df_county = df[df["county"]==point_county]
    distance_all = df_county.apply(lambda x : distance(point[0],point[1],x['latitude'],x['longitude']),axis=1)
    #print(distance_all)
    places_nearby_50 = list(filter(lambda x: x <= 50.0, distance_all.to_numpy()))#distance_all[distance_all>=50 and distance_all!=0]
    nearby.append(len(places_nearby_50))

data['events_within_50kms'] = nearby
#print(data['places_within_50kms'][0:10])

data.to_csv("assignment_output.csv")
