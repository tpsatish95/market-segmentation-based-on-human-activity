from collections import defaultdict
from sklearn import cluster
import matplotlib.pyplot as plt
import random
import operator

gps = []
X_Train = []
y_Train = []
X_Test = []
y_Test = []
userActvity = defaultdict(list)
actionMap = {"1" : "WALKING",
			 "2" : "WALKING_UPSTAIRS",
             "3" : "WALKING_DOWNSTAIRS",
             "4" : "SITTING",
             "5" : "STANDING",
             "6" : "LAYING"}

with open("../train/subject_train.txt") as train_file:
	X_Train = train_file.readlines()
	X_Train = [x.strip() for x in X_Train]

with open("../train/y_train.txt") as train_file:
	y_Train = train_file.readlines()
	y_Train = [int(y.strip()) for y in y_Train]

for i,j in zip(X_Train,y_Train):
	userActvity[i].append(j)

with open("../test/subject_test.txt") as test_file:
	X_Test = test_file.readlines()
	X_Test = [x.strip() for x in X_Test]

with open("../test/y_test.txt") as test_file:
	y_Test = test_file.readlines()
	y_Test = [int(y.strip()) for y in y_Test]

for i,j in zip(X_Test,y_Test):
	userActvity[i].append(j)

# to normalize userActivity lenghts
lens= []
for jk in userActvity.keys():
	lens.append(len(userActvity[jk]))

for jk in userActvity.keys():
	userActvity[jk] = userActvity[jk][:min(lens)]
###

# generating gps co-ords randomly
gps = [(random.randrange(0,100), random.randrange(0,100)) for i in range(len(userActvity))] 

# find best cluster
# K-Means
# for i in range(2,len(userActvity)):
# 	kmeans = cluster.KMeans(n_clusters=i).fit(gps)
# 	print(str(i) + " -> " + str(kmeans.inertia_))

# K_Means
kmeans = cluster.KMeans(n_clusters=5).fit(gps)
# gps based user to region mapping - [labels]
labels = kmeans.labels_

color_map = {0 :"r",1 :"g",2 :"b",3 :"k",4 :"y"}
label_color = [color_map[i] for i in labels]
gps_x = [g[0] for g in gps]
gps_y = [g[1] for g in gps]
plt.scatter(gps_x,gps_y,c = label_color)
plt.savefig('kmeansGPS.png')

region2UserActivities = defaultdict(list)

for i in range(1,len(userActvity)+1):
	region2UserActivities[labels[i-1]].append([gps[i-1],userActvity[str(i)]])

for k in region2UserActivities.keys():
	X = []
	gpsRe = []
	for j in region2UserActivities[k]:
		X.append(j[1])
		gpsRe.append(j[0])

	kmeans = cluster.KMeans(n_clusters=2).fit(X)
	labels = kmeans.labels_
	print( "Region " + str(k))
	
	color_map = {0 :"k",1 :"w"}
	label_color = [color_map[i] for i in labels]
	gps_x = [g[0] for g in gpsRe]
	gps_y = [g[1] for g in gpsRe]
	plt.figure(k+2)
	plt.scatter(gps_x,gps_y,c = label_color)
	plt.savefig('kmeansGPS_'+str(k)+'.png')

	countEachActivityPerRegion = defaultdict()
	for r in range(2):
		countEachActivityPerRegion[r] = defaultdict(int)
	
	for instance,l in zip(X,labels):
		for ins in instance:
			countEachActivityPerRegion[l][ins] += 1

	for r in range(2):
		print("Area "+str(r)+" -> " + str(countEachActivityPerRegion[r]))
		top3 = sorted(countEachActivityPerRegion[r].items(), key = operator.itemgetter(1), reverse = True)[:3]
		print("Top3 Actvities")
		for t in top3:
			print(str(actionMap[str(t[0])]) + " -> " + str(t[1]))