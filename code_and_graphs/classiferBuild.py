from __future__ import print_function

import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn import cluster, datasets, preprocessing

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


X_Train = []
y_Train = []
X_Test = []
y_Test = []

with open("../train/X_train.txt") as train_file:
	X_Train = train_file.readlines()
	X_Train = [line.split() for line in X_Train]
	for i in range(len(X_Train)):
		X_Train[i] = [float(j) for j in X_Train[i]]

with open("../train/y_train.txt") as train_file:
	y_Train = train_file.readlines()
	y_Train = [y.strip() for y in y_Train]

with open("../test/X_test.txt") as test_file:
	X_Test = test_file.readlines()
	X_Test = [line.split() for line in X_Test]
	for i in range(len(X_Test)):
		X_Test[i] = [float(j) for j in X_Test[i]]

with open("../test/y_test.txt") as test_file:
	y_Test = test_file.readlines()
	y_Test = [y.strip() for y in y_Test]

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_Train, y_Train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_Test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_Test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_Test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_Test, pred))

    print()
    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time,clf

# Analyze various algos
results = []

for clf, name in ((RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")
        ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# # Train sparse Naive Bayes classifiers
# print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# make some plots
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.savefig('classifierComparisons.png')

# model build ()
LSVC = benchmark(LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3))
# Save Model
save_obj(LSVC[4],"classifier")


# Clustering Experiments
# # K-Means
# t0 = time()
# kmeans = cluster.KMeans(n_clusters=6, n_init = 300).fit(X_Train)
# print(str(time()-t0))

# save_obj(kmeans,"kmeans")

# knn = benchmark(KNeighborsClassifier(n_neighbors=10))[4]
# save_obj(knn,"cluster")

# knn = load_obj("cluster")
# print(knn.predict(np.array([2.5717778e-001,-2.3285230e-002,-1.4653762e-002,-9.3840400e-001,-9.2009078e-001,-6.6768331e-001,-9.5250112e-001,-9.2524867e-001,-6.7430222e-001,-8.9408755e-001,-5.5457721e-001,-4.6622295e-001,7.1720847e-001,6.3550240e-001,7.8949666e-001,-8.7776423e-001,-9.9776606e-001,-9.9841381e-001,-9.3434525e-001,-9.7566897e-001,-9.4982365e-001,-8.3047780e-001,-1.6808416e-001,-3.7899553e-001,2.4621698e-001,5.2120364e-001,-4.8779311e-001,4.8228047e-001,-4.5462113e-002,2.1195505e-001,-1.3489443e-001,1.3085848e-001,-1.4176313e-002,-1.0597085e-001,7.3544013e-002,-1.7151642e-001,4.0062978e-002,7.6988933e-002,-4.9054573e-001,-7.0900265e-001,9.3648925e-001,-2.8271916e-001,1.1528825e-001,-9.2542727e-001,-9.3701413e-001,-5.6428842e-001,-9.3001992e-001,-9.3782195e-001,-6.0558770e-001,9.0608259e-001,-2.7924413e-001,1.5289519e-001,9.4446140e-001,-2.6215956e-001,-7.6161676e-002,-1.7826920e-002,8.2929682e-001,-8.6462060e-001,-9.6779531e-001,-9.4972666e-001,-9.4611920e-001,-7.5971815e-001,-4.2497535e-001,-1.0000000e+000,2.1922731e-001,-4.3025357e-001,4.3104828e-001,-4.3183892e-001,4.3277380e-001,-7.9546772e-001,7.8131389e-001,-7.8039147e-001,7.8527158e-001,-9.8441024e-001,9.8717986e-001,-9.8941477e-001,9.8768613e-001,9.8058028e-001,-9.9635177e-001,-9.6011706e-001,7.2046007e-002,4.5754401e-002,-1.0604266e-001,-9.0668276e-001,-9.3801639e-001,-9.3593583e-001,-9.1608093e-001,-9.3672546e-001,-9.4905379e-001,-9.0322415e-001,-9.4981833e-001,-8.9140347e-001,8.9847935e-001,9.5018164e-001,9.4615279e-001,-9.3067288e-001,-9.9504593e-001,-9.9749551e-001,-9.9701560e-001,-9.3641600e-001,-9.4687413e-001,-9.6877461e-001,-8.5174151e-002,-3.1026304e-001,-5.1028758e-001,5.2148173e-001,-2.2588966e-001,4.9172843e-001,3.1275555e-001,2.2979680e-001,1.1395925e-001,2.1987861e-001,4.2297454e-001,-8.2633177e-002,1.4042653e-001,-1.9623228e-001,7.2357939e-002,-2.6486023e-001,3.5852150e-002,-3.4973525e-001,1.1997616e-001,-9.1792335e-002,1.8962854e-001,-8.8308911e-001,-8.1616360e-001,-9.4088123e-001,-8.8861231e-001,-8.5780102e-001,-9.4581827e-001,-6.6341057e-001,-7.1343663e-001,-6.4867861e-001,8.3710039e-001,8.2525677e-001,8.1097714e-001,-7.9649994e-001,-9.7961636e-001,-9.8290006e-001,-9.9403684e-001,-8.8655788e-001,-9.0610426e-001,-9.5804876e-001,7.7403279e-001,-2.6770588e-001,4.5224806e-001,-7.8451267e-002,-1.2578616e-002,2.3598156e-001,-1.9904751e-001,3.3917840e-002,-8.0780533e-002,6.9987153e-003,2.4488551e-001,2.1651661e-001,-2.7968077e-001,2.4973875e-001,1.7719752e-002,6.4846454e-001,-2.3693109e-001,-3.0173469e-001,-2.0489621e-001,-1.7448771e-001,-9.3389340e-002,-9.0122415e-001,-9.1086005e-001,-9.3925042e-001,-9.1036271e-001,-9.2735675e-001,-9.5355413e-001,-8.6791431e-001,-9.1349778e-001,-8.9757791e-001,9.0493669e-001,9.1730839e-001,9.4761220e-001,-9.2960905e-001,-9.9468622e-001,-9.9579057e-001,-9.9781265e-001,-9.3654081e-001,-9.5887957e-001,-9.7034831e-001,3.6619120e-002,7.6459933e-002,-1.9712605e-001,1.0651426e-001,-2.0811895e-002,1.9325784e-001,3.0447875e-001,1.1572923e-001,5.4149600e-002,6.8951237e-002,1.9704960e-001,3.0992826e-001,-2.1265711e-001,1.7317814e-001,1.4584454e-001,1.2400875e-001,-1.5534634e-001,-3.2343727e-001,-8.6692938e-001,-7.0519112e-001,-7.4402172e-001,-7.6079564e-001,-9.8164870e-001,-8.6692938e-001,-9.8016578e-001,-8.5947423e-001,2.5510436e-001,5.3779695e-002,-2.0414449e-001,6.1052755e-001,-5.6444932e-001,-8.6692938e-001,-7.0519112e-001,-7.4402172e-001,-7.6079564e-001,-9.8164870e-001,-8.6692938e-001,-9.8016578e-001,-8.5947423e-001,2.5510436e-001,5.3779695e-002,-2.0414449e-001,6.1052755e-001,-5.6444932e-001,-9.2976655e-001,-8.9599425e-001,-9.0041731e-001,-9.0300439e-001,-9.7501109e-001,-9.2976655e-001,-9.9560772e-001,-9.1412066e-001,-1.2955231e-001,2.3891093e-001,-3.4559715e-001,3.2646236e-001,-2.6304800e-001,-7.9554393e-001,-7.6207322e-001,-7.8267232e-001,-7.1659365e-001,-7.6419261e-001,-7.9554393e-001,-9.7415212e-001,-8.3958101e-001,6.6756269e-001,3.5621137e-002,-1.6189398e-001,1.5325006e-001,-6.7596040e-003,-9.2519489e-001,-8.9434361e-001,-9.0014668e-001,-9.1673708e-001,-9.7636665e-001,-9.2519489e-001,-9.9582422e-001,-9.1183750e-001,3.3165431e-001,5.1695316e-001,-5.1350400e-001,4.1319806e-002,1.1835012e-002,-9.1850969e-001,-9.1821319e-001,-7.8909145e-001,-9.4829035e-001,-9.2513687e-001,-6.3631674e-001,-9.3068029e-001,-9.2443848e-001,-7.2490255e-001,-9.6842407e-001,-9.4013675e-001,-5.9718873e-001,-9.6613713e-001,-9.8445054e-001,-9.5208707e-001,-8.6506318e-001,-9.9784371e-001,-9.9603625e-001,-9.4019503e-001,-9.0484043e-001,-9.3381245e-001,-8.6938291e-001,-3.3967327e-001,-4.8580324e-001,-1.6625765e-001,-1.0000000e+000,-1.0000000e+000,-1.0000000e+000,1.1116947e-002,1.2125069e-001,-5.2294869e-001,-5.7199950e-001,-8.9461236e-001,-3.3826592e-001,-6.8679745e-001,1.8955250e-001,-1.1359571e-001,-9.9850627e-001,-9.9794262e-001,-9.9559788e-001,-9.9503559e-001,-9.9597690e-001,-9.9148743e-001,-9.9213116e-001,-9.9977512e-001,-9.9820395e-001,-9.9474673e-001,-9.9430903e-001,-9.9469323e-001,-9.9802046e-001,-9.9372747e-001,-9.9620895e-001,-9.9797615e-001,-9.9892091e-001,-9.9640106e-001,-9.9755992e-001,-9.9506803e-001,-9.9735126e-001,-9.9823805e-001,-9.9607053e-001,-9.9792459e-001,-9.9635664e-001,-9.9763980e-001,-9.9622004e-001,-9.9622539e-001,-9.2470550e-001,-9.9409728e-001,-9.9532347e-001,-9.9896944e-001,-9.9790162e-001,-9.9492962e-001,-9.8926072e-001,-9.8008540e-001,-9.3793669e-001,-9.9667757e-001,-9.9702540e-001,-9.8642738e-001,-9.3920079e-001,-9.9843184e-001,-8.9963316e-001,-9.3748500e-001,-9.2355140e-001,-9.2442913e-001,-9.4321038e-001,-9.4789152e-001,-8.9661455e-001,-9.3830911e-001,-9.4257570e-001,-9.4863426e-001,-9.5832542e-001,-9.5881686e-001,-9.4388241e-001,-9.8730333e-001,-9.7846559e-001,-9.0527425e-001,-9.9503609e-001,-9.9749931e-001,-9.9703070e-001,-8.8707736e-001,-9.3581956e-001,-9.5365327e-001,-4.7066160e-001,-6.7217180e-001,-5.9627404e-001,-5.2000000e-001,8.0000000e-002,3.2000000e-001,4.5100539e-001,1.3716703e-001,-1.8029913e-001,-5.8008614e-001,-9.0807003e-001,-6.2552686e-001,-9.4275716e-001,-6.6191006e-001,-9.1153211e-001,-9.9901148e-001,-9.9763006e-001,-9.9598477e-001,-9.9470846e-001,-9.9557158e-001,-9.8787651e-001,-9.8644523e-001,-9.9820517e-001,-9.9804041e-001,-9.9449682e-001,-9.9204962e-001,-9.8633303e-001,-9.9689353e-001,-9.9076269e-001,-9.9940916e-001,-9.9864806e-001,-9.9874875e-001,-9.9640627e-001,-9.9717222e-001,-9.9453020e-001,-9.9791979e-001,-9.9996970e-001,-9.9864167e-001,-9.9740170e-001,-9.9541085e-001,-9.9818240e-001,-9.9846306e-001,-9.9599581e-001,-9.9363885e-001,-9.9783780e-001,-9.9712931e-001,-9.9874950e-001,-9.9670876e-001,-9.9510966e-001,-9.9848581e-001,-9.9909354e-001,-9.9592463e-001,-9.9795094e-001,-9.9601290e-001,-9.9846008e-001,-9.9626705e-001,-9.9770046e-001,-8.2355788e-001,-8.0791598e-001,-9.1791256e-001,-9.0326274e-001,-8.2267700e-001,-9.5616508e-001,-8.6512704e-001,-8.3180082e-001,-9.4105617e-001,-9.0479801e-001,-8.7925663e-001,-9.6778849e-001,-8.7859934e-001,-9.4832915e-001,-9.1968517e-001,-8.2847203e-001,-9.9294953e-001,-9.8266311e-001,-9.9799327e-001,-8.7889440e-001,-8.3815197e-001,-9.2913995e-001,7.5814806e-004,2.0014368e-001,-2.5338416e-001,-1.0000000e+000,-9.3548387e-001,-9.3103448e-001,1.8403457e-001,-5.9322857e-002,4.3810716e-001,-3.9542276e-001,-6.9876160e-001,-3.8745724e-001,-7.8639417e-001,-4.8565359e-001,-7.8681512e-001,-9.9462908e-001,-9.9046914e-001,-9.9277782e-001,-9.9567969e-001,-9.8707772e-001,-9.8665097e-001,-9.8411445e-001,-9.8525021e-001,-9.9355255e-001,-9.9243025e-001,-9.8560582e-001,-9.8461702e-001,-9.9326921e-001,-9.9272545e-001,-9.7792782e-001,-9.9485223e-001,-9.9781915e-001,-9.9484295e-001,-9.9240991e-001,-9.8812267e-001,-9.9043323e-001,-9.8796127e-001,-9.8179096e-001,-9.9639919e-001,-9.9145229e-001,-9.8805999e-001,-9.8204835e-001,-9.9337999e-001,-9.9887769e-001,-9.9836665e-001,-9.9846730e-001,-9.9837462e-001,-9.9890591e-001,-9.9589361e-001,-9.9312836e-001,-9.9547231e-001,-9.9838710e-001,-9.9775430e-001,-9.9809593e-001,-9.9414765e-001,-9.9819058e-001,-9.9829005e-001,-7.9094643e-001,-7.1107400e-001,-7.2670699e-001,-7.7769715e-001,-9.4488134e-001,-7.9094643e-001,-9.5398356e-001,-8.7354261e-001,-1.7459288e-001,-1.0000000e+000,-4.8345254e-001,1.1040681e-002,-3.8451662e-001,-8.9506118e-001,-8.9635958e-001,-8.8819740e-001,-9.2846566e-001,-8.9809981e-001,-8.9506118e-001,-9.9347143e-001,-9.2147669e-001,-4.8461929e-001,-1.0000000e+000,-3.5355792e-002,-2.5424830e-001,-7.0032573e-001,-7.7061000e-001,-7.9711285e-001,-7.6448457e-001,-8.2018760e-001,-9.3795935e-001,-7.7061000e-001,-9.7095802e-001,-7.9838652e-001,1.7943523e-001,-1.0000000e+000,-4.7391298e-002,-4.6784901e-001,-7.6132577e-001,-8.9016545e-001,-9.0730756e-001,-8.9530057e-001,-9.1788296e-001,-9.0982876e-001,-8.9016545e-001,-9.9410543e-001,-8.9802151e-001,-2.3481529e-001,-1.0000000e+000,7.1645446e-002,-3.3037044e-001,-7.0597388e-001,6.4624029e-003,1.6291982e-001,-8.2588562e-001,2.7115145e-001,-7.2000927e-001,2.7680104e-001,-5.7978304e-002])))
# print(knn.predict(np.array([2.7225459e-001,-2.5051205e-002,-1.3318432e-001,-9.9115349e-001,-9.6354868e-001,-9.7316133e-001,-9.9180349e-001,-9.6257616e-001,-9.7089159e-001,-9.3442560e-001,-5.5771311e-001,-8.2518126e-001,8.4261103e-001,6.7544459e-001,8.2567640e-001,-9.7181292e-001,-9.9991767e-001,-9.9949506e-001,-9.9839174e-001,-9.9288641e-001,-9.6863812e-001,-9.6852005e-001,-5.8500907e-001,-6.3746941e-001,-7.9889559e-001,1.6548406e-001,-5.8500401e-002,6.8108751e-002,4.4890429e-002,2.0459645e-002,-8.9055644e-002,2.9905697e-001,-1.9888183e-001,-6.4258325e-002,9.3879381e-002,1.4835327e-002,-2.9779648e-001,1.4360917e-001,1.1356622e-001,6.0136772e-001,8.8833105e-001,-4.1194179e-001,7.5492335e-002,-9.9607584e-001,-9.9233821e-001,-9.6880022e-001,-9.9588612e-001,-9.9360962e-001,-9.7187781e-001,8.1524240e-001,-4.2735611e-001,7.5724798e-002,9.0885565e-001,-3.8394094e-001,6.8691314e-002,1.0986822e-001,7.0633984e-001,-7.0073490e-001,-9.9030176e-001,-9.9579551e-001,-9.9746797e-001,-9.7746019e-001,-1.0000000e+000,-1.0000000e+000,-2.8313533e-001,-4.2681194e-001,4.5291422e-001,-4.7912003e-001,5.0552340e-001,-4.0264644e-001,3.5949949e-001,-3.5511795e-001,3.6795785e-001,-4.8652263e-001,4.9363618e-001,-5.0061515e-001,5.0485230e-001,4.3962624e-001,9.2562175e-001,6.9906794e-001,7.2322420e-002,-4.0930707e-003,2.2683017e-003,-9.8679902e-001,-9.7633977e-001,-9.8315362e-001,-9.8577659e-001,-9.7164622e-001,-9.8157377e-001,-9.8586036e-001,-9.9045215e-001,-9.8675837e-001,9.9217091e-001,9.8327498e-001,9.8413313e-001,-9.8177966e-001,-9.9982335e-001,-9.9949371e-001,-9.9961105e-001,-9.8014789e-001,-9.6999907e-001,-9.8273651e-001,-6.1277580e-001,-5.9222739e-001,-6.3488699e-001,1.7752853e-001,1.5743522e-001,2.5988588e-001,1.1705944e-001,6.4384613e-002,-6.7370380e-002,2.7361225e-001,3.6234954e-001,-5.5685239e-005,1.5665882e-001,1.3743915e-001,-2.5396967e-004,-1.7094173e-001,3.3747703e-001,3.0664537e-001,-3.2800503e-002,-9.0070366e-002,1.1430315e-001,-9.7998548e-001,-9.8331456e-001,-9.7736636e-001,-9.7927729e-001,-9.8437200e-001,-9.7905512e-001,-8.7068136e-001,-9.4428039e-001,-7.3525600e-001,8.3317627e-001,9.0069575e-001,8.1791186e-001,-9.7365609e-001,-9.9976666e-001,-9.9977800e-001,-9.9936968e-001,-9.7540070e-001,-9.8618819e-001,-9.8434324e-001,-3.5577012e-001,-5.7781716e-001,5.4792030e-002,-1.6576050e-001,2.6218317e-003,2.0897571e-001,8.7332174e-002,-9.1461528e-002,8.2965428e-002,1.1980126e-001,-4.5105405e-002,-4.6956943e-003,-1.0355955e-001,1.0943182e-001,1.8562141e-001,-8.9016640e-002,8.0849651e-002,-9.8726937e-002,-9.8093802e-002,-3.1359710e-002,-5.1153105e-002,-9.8633375e-001,-9.8639137e-001,-9.8506684e-001,-9.8541630e-001,-9.8811150e-001,-9.8794292e-001,-9.8767957e-001,-9.8306771e-001,-9.6138923e-001,9.8658421e-001,9.8713669e-001,9.8931350e-001,-9.8804575e-001,-9.9985252e-001,-9.9987437e-001,-9.9980357e-001,-9.8404397e-001,-9.8995608e-001,-9.9014952e-001,-4.3819442e-001,-3.2341547e-001,-5.1094188e-001,-3.6176453e-002,-1.7747149e-001,1.4331696e-001,4.4858521e-002,-3.9255220e-002,1.2364810e-001,3.0676668e-001,-1.2246685e-001,5.5444014e-002,-1.2975553e-001,-1.6416618e-002,1.7697377e-001,4.0635664e-001,8.7950255e-003,-4.1378413e-001,-9.7278247e-001,-9.7340407e-001,-9.7442408e-001,-9.7557291e-001,-9.9121254e-001,-9.7278247e-001,-9.9926938e-001,-9.7514144e-001,-3.7741593e-001,-5.5051719e-002,4.8763686e-002,-7.3980457e-002,5.2923916e-002,-9.7278247e-001,-9.7340407e-001,-9.7442408e-001,-9.7557291e-001,-9.9121254e-001,-9.7278247e-001,-9.9926938e-001,-9.7514144e-001,-3.7741593e-001,-5.5051719e-002,4.8763686e-002,-7.3980457e-002,5.2923916e-002,-9.8230467e-001,-9.8736749e-001,-9.8776768e-001,-9.8845521e-001,-9.8510938e-001,-9.8230467e-001,-9.9964071e-001,-9.8974462e-001,-7.4951281e-001,4.2149466e-001,-8.9567127e-002,-3.3289524e-001,-2.3370434e-001,-9.7418211e-001,-9.8141963e-001,-9.7908712e-001,-9.8268288e-001,-9.8000317e-001,-9.7418211e-001,-9.9956478e-001,-9.7732371e-001,-8.2891711e-002,2.3779314e-001,-3.5065703e-001,9.5391314e-002,2.3196730e-001,-9.8801891e-001,-9.8567978e-001,-9.8738357e-001,-9.8460963e-001,-9.9626916e-001,-9.8801891e-001,-9.9986286e-001,-9.8826605e-001,-3.9880550e-001,6.6708295e-001,-4.6629777e-001,-2.3797414e-001,2.8145829e-002,-9.8861582e-001,-9.6663407e-001,-9.7627432e-001,-9.9236991e-001,-9.6315009e-001,-9.7245528e-001,-9.8962291e-001,-9.6719116e-001,-9.7143094e-001,-9.9462169e-001,-9.6489543e-001,-9.6705081e-001,-9.8946646e-001,-9.9176127e-001,-9.9873112e-001,-9.7909606e-001,-9.9992259e-001,-9.9897608e-001,-9.9920169e-001,-9.8536560e-001,-9.8050813e-001,-9.6437831e-001,-9.4635692e-001,-7.0966494e-001,-7.2952251e-001,-6.7741935e-001,-1.0000000e+000,-1.0000000e+000,2.7733679e-002,1.8259313e-002,-6.0243493e-003,-5.6608392e-001,-8.9196664e-001,-9.9655594e-002,-4.3432918e-001,-3.2623971e-001,-5.4560670e-001,-9.9995259e-001,-9.9992014e-001,-9.9972258e-001,-9.9980783e-001,-9.9980791e-001,-9.9982107e-001,-9.9990862e-001,-9.9998406e-001,-9.9994099e-001,-9.9971062e-001,-9.9982791e-001,-9.9993392e-001,-9.9992721e-001,-9.9978881e-001,-9.9884670e-001,-9.9965652e-001,-9.9961245e-001,-9.9961698e-001,-9.9948413e-001,-9.9918238e-001,-9.9960223e-001,-9.9966273e-001,-9.9895617e-001,-9.9952237e-001,-9.9934182e-001,-9.9961519e-001,-9.9897916e-001,-9.9953329e-001,-9.9921658e-001,-9.9953502e-001,-9.9952437e-001,-9.9974323e-001,-9.9976337e-001,-9.9971642e-001,-9.9957939e-001,-9.9999569e-001,-9.9925506e-001,-9.9963702e-001,-9.9978014e-001,-9.9970980e-001,-9.9920265e-001,-9.9977052e-001,-9.8641512e-001,-9.7693686e-001,-9.8054747e-001,-9.8847165e-001,-9.7725064e-001,-9.8430666e-001,-9.8564740e-001,-9.7877817e-001,-9.8274249e-001,-9.9137416e-001,-9.7742498e-001,-9.8752830e-001,-9.9872255e-001,-9.9540666e-001,-9.7444866e-001,-9.8083550e-001,-9.9982305e-001,-9.9949440e-001,-9.9961103e-001,-9.8789503e-001,-9.8377836e-001,-9.8078332e-001,-1.0000000e+000,-9.3912029e-001,-1.0000000e+000,-1.2000000e-001,-5.2000000e-001,-1.6000000e-001,2.3514688e-001,-1.5969842e-001,-9.5183692e-002,-5.2947165e-001,-8.4939943e-001,-4.1171051e-001,-8.0273438e-001,-5.5795355e-001,-8.9214805e-001,-9.9996651e-001,-9.9994276e-001,-9.9972337e-001,-9.9979658e-001,-9.9977635e-001,-9.9973390e-001,-9.9986501e-001,-9.9999922e-001,-9.9994973e-001,-9.9970147e-001,-9.9975404e-001,-9.9986723e-001,-9.9985804e-001,-9.9969657e-001,-9.9946446e-001,-9.9967977e-001,-9.9955790e-001,-9.9968665e-001,-9.9958072e-001,-9.9927653e-001,-9.9947835e-001,-9.9999591e-001,-9.9959032e-001,-9.9953876e-001,-9.9939725e-001,-9.9954765e-001,-9.9950938e-001,-9.9959281e-001,-9.9953044e-001,-9.9948196e-001,-9.9948368e-001,-9.9972592e-001,-9.9976020e-001,-9.9971648e-001,-9.9922690e-001,-9.9981753e-001,-9.9942335e-001,-9.9963162e-001,-9.9973892e-001,-9.9924619e-001,-9.9943175e-001,-9.9975713e-001,-9.7913434e-001,-9.8077626e-001,-9.7611518e-001,-9.8029393e-001,-9.8515793e-001,-9.7972877e-001,-9.7570482e-001,-9.8270225e-001,-9.7472749e-001,-9.8452920e-001,-9.9197636e-001,-9.8507366e-001,-9.9912320e-001,-9.9752417e-001,-9.9731872e-001,-9.7886550e-001,-9.9976113e-001,-9.9981742e-001,-9.9964299e-001,-9.8926545e-001,-9.7761750e-001,-9.8290848e-001,-6.5440007e-001,-5.7286044e-001,-6.3715945e-001,-8.6666667e-001,-9.3548387e-001,-7.2413793e-001,-4.2225647e-001,8.0033882e-002,-1.7980592e-001,-3.6837210e-001,-7.6086777e-001,-6.8799022e-001,-9.5015015e-001,-4.0084359e-001,-7.4160223e-001,-9.9975881e-001,-9.9991589e-001,-9.9994395e-001,-9.9993625e-001,-9.9996200e-001,-9.9987123e-001,-9.9994344e-001,-9.9999690e-001,-9.9976096e-001,-9.9993098e-001,-9.9992262e-001,-9.9996711e-001,-9.9976229e-001,-9.9993257e-001,-9.9984847e-001,-9.9992415e-001,-9.9990215e-001,-9.9991726e-001,-9.9997496e-001,-9.9981896e-001,-9.9970392e-001,-9.9985429e-001,-9.9984945e-001,-9.9988367e-001,-9.9994308e-001,-9.9972882e-001,-9.9981384e-001,-9.9991943e-001,-9.9969314e-001,-9.9984535e-001,-9.9992690e-001,-9.9990887e-001,-9.9994717e-001,-9.9969881e-001,-9.9988488e-001,-9.9999119e-001,-9.9965825e-001,-9.9988885e-001,-9.9988593e-001,-9.9993131e-001,-9.9965187e-001,-9.9990366e-001,-9.7445530e-001,-9.7575665e-001,-9.7035080e-001,-9.8022231e-001,-9.9169490e-001,-9.7445530e-001,-9.9937852e-001,-9.7575388e-001,-7.6917080e-001,-1.0000000e+000,-4.7700824e-003,-4.2081578e-001,-6.9378174e-001,-9.8616213e-001,-9.8789437e-001,-9.8603047e-001,-9.9110134e-001,-9.9171955e-001,-9.8616213e-001,-9.9977779e-001,-9.8851977e-001,-1.0000000e+000,-1.0000000e+000,4.4725853e-001,-5.1512642e-001,-8.5285036e-001,-9.8417080e-001,-9.8245605e-001,-9.8239648e-001,-9.8518025e-001,-9.9091964e-001,-9.8417080e-001,-9.9976247e-001,-9.8829140e-001,-7.3760865e-001,-8.4615385e-001,-4.4412803e-002,-3.4167689e-001,-7.1941435e-001,-9.8489458e-001,-9.8743883e-001,-9.8542920e-001,-9.8939759e-001,-9.8264752e-001,-9.8489458e-001,-9.9985412e-001,-9.8215772e-001,-8.2873420e-001,-9.6825397e-001,3.1134412e-001,-4.4096996e-001,-7.6582321e-001,5.5357260e-003,-3.6520149e-002,5.8946399e-002,2.1627788e-001,-6.0405909e-001,3.6732106e-001,-3.0169870e-002])))