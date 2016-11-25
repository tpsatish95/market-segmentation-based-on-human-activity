import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

class predictor(object):

	def __init__(self):
		self.classifier = load_obj("classifier")
		self.actionMap = {"1" : "WALKING",
		 				  "2" : "WALKING_UPSTAIRS",
		                  "3" : "WALKING_DOWNSTAIRS",
		                  "4" : "SITTING",
		                  "5" : "STANDING",
		                  "6" : "LAYING"}
	def get(self,inputX,encoded = 1):
		if encoded == 1:
			return self.classifier.predict(inputX)[0]
		elif encoded == 0:
			return self.actionMap[self.classifier.predict(inputX)[0]]


# p = predictor()
# print(p.get([2.5717778e-001,-2.3285230e-002,-1.4653762e-002,-9.3840400e-001,-9.2009078e-001,-6.6768331e-001,-9.5250112e-001,-9.2524867e-001,-6.7430222e-001,-8.9408755e-001,-5.5457721e-001,-4.6622295e-001,7.1720847e-001,6.3550240e-001,7.8949666e-001,-8.7776423e-001,-9.9776606e-001,-9.9841381e-001,-9.3434525e-001,-9.7566897e-001,-9.4982365e-001,-8.3047780e-001,-1.6808416e-001,-3.7899553e-001,2.4621698e-001,5.2120364e-001,-4.8779311e-001,4.8228047e-001,-4.5462113e-002,2.1195505e-001,-1.3489443e-001,1.3085848e-001,-1.4176313e-002,-1.0597085e-001,7.3544013e-002,-1.7151642e-001,4.0062978e-002,7.6988933e-002,-4.9054573e-001,-7.0900265e-001,9.3648925e-001,-2.8271916e-001,1.1528825e-001,-9.2542727e-001,-9.3701413e-001,-5.6428842e-001,-9.3001992e-001,-9.3782195e-001,-6.0558770e-001,9.0608259e-001,-2.7924413e-001,1.5289519e-001,9.4446140e-001,-2.6215956e-001,-7.6161676e-002,-1.7826920e-002,8.2929682e-001,-8.6462060e-001,-9.6779531e-001,-9.4972666e-001,-9.4611920e-001,-7.5971815e-001,-4.2497535e-001,-1.0000000e+000,2.1922731e-001,-4.3025357e-001,4.3104828e-001,-4.3183892e-001,4.3277380e-001,-7.9546772e-001,7.8131389e-001,-7.8039147e-001,7.8527158e-001,-9.8441024e-001,9.8717986e-001,-9.8941477e-001,9.8768613e-001,9.8058028e-001,-9.9635177e-001,-9.6011706e-001,7.2046007e-002,4.5754401e-002,-1.0604266e-001,-9.0668276e-001,-9.3801639e-001,-9.3593583e-001,-9.1608093e-001,-9.3672546e-001,-9.4905379e-001,-9.0322415e-001,-9.4981833e-001,-8.9140347e-001,8.9847935e-001,9.5018164e-001,9.4615279e-001,-9.3067288e-001,-9.9504593e-001,-9.9749551e-001,-9.9701560e-001,-9.3641600e-001,-9.4687413e-001,-9.6877461e-001,-8.5174151e-002,-3.1026304e-001,-5.1028758e-001,5.2148173e-001,-2.2588966e-001,4.9172843e-001,3.1275555e-001,2.2979680e-001,1.1395925e-001,2.1987861e-001,4.2297454e-001,-8.2633177e-002,1.4042653e-001,-1.9623228e-001,7.2357939e-002,-2.6486023e-001,3.5852150e-002,-3.4973525e-001,1.1997616e-001,-9.1792335e-002,1.8962854e-001,-8.8308911e-001,-8.1616360e-001,-9.4088123e-001,-8.8861231e-001,-8.5780102e-001,-9.4581827e-001,-6.6341057e-001,-7.1343663e-001,-6.4867861e-001,8.3710039e-001,8.2525677e-001,8.1097714e-001,-7.9649994e-001,-9.7961636e-001,-9.8290006e-001,-9.9403684e-001,-8.8655788e-001,-9.0610426e-001,-9.5804876e-001,7.7403279e-001,-2.6770588e-001,4.5224806e-001,-7.8451267e-002,-1.2578616e-002,2.3598156e-001,-1.9904751e-001,3.3917840e-002,-8.0780533e-002,6.9987153e-003,2.4488551e-001,2.1651661e-001,-2.7968077e-001,2.4973875e-001,1.7719752e-002,6.4846454e-001,-2.3693109e-001,-3.0173469e-001,-2.0489621e-001,-1.7448771e-001,-9.3389340e-002,-9.0122415e-001,-9.1086005e-001,-9.3925042e-001,-9.1036271e-001,-9.2735675e-001,-9.5355413e-001,-8.6791431e-001,-9.1349778e-001,-8.9757791e-001,9.0493669e-001,9.1730839e-001,9.4761220e-001,-9.2960905e-001,-9.9468622e-001,-9.9579057e-001,-9.9781265e-001,-9.3654081e-001,-9.5887957e-001,-9.7034831e-001,3.6619120e-002,7.6459933e-002,-1.9712605e-001,1.0651426e-001,-2.0811895e-002,1.9325784e-001,3.0447875e-001,1.1572923e-001,5.4149600e-002,6.8951237e-002,1.9704960e-001,3.0992826e-001,-2.1265711e-001,1.7317814e-001,1.4584454e-001,1.2400875e-001,-1.5534634e-001,-3.2343727e-001,-8.6692938e-001,-7.0519112e-001,-7.4402172e-001,-7.6079564e-001,-9.8164870e-001,-8.6692938e-001,-9.8016578e-001,-8.5947423e-001,2.5510436e-001,5.3779695e-002,-2.0414449e-001,6.1052755e-001,-5.6444932e-001,-8.6692938e-001,-7.0519112e-001,-7.4402172e-001,-7.6079564e-001,-9.8164870e-001,-8.6692938e-001,-9.8016578e-001,-8.5947423e-001,2.5510436e-001,5.3779695e-002,-2.0414449e-001,6.1052755e-001,-5.6444932e-001,-9.2976655e-001,-8.9599425e-001,-9.0041731e-001,-9.0300439e-001,-9.7501109e-001,-9.2976655e-001,-9.9560772e-001,-9.1412066e-001,-1.2955231e-001,2.3891093e-001,-3.4559715e-001,3.2646236e-001,-2.6304800e-001,-7.9554393e-001,-7.6207322e-001,-7.8267232e-001,-7.1659365e-001,-7.6419261e-001,-7.9554393e-001,-9.7415212e-001,-8.3958101e-001,6.6756269e-001,3.5621137e-002,-1.6189398e-001,1.5325006e-001,-6.7596040e-003,-9.2519489e-001,-8.9434361e-001,-9.0014668e-001,-9.1673708e-001,-9.7636665e-001,-9.2519489e-001,-9.9582422e-001,-9.1183750e-001,3.3165431e-001,5.1695316e-001,-5.1350400e-001,4.1319806e-002,1.1835012e-002,-9.1850969e-001,-9.1821319e-001,-7.8909145e-001,-9.4829035e-001,-9.2513687e-001,-6.3631674e-001,-9.3068029e-001,-9.2443848e-001,-7.2490255e-001,-9.6842407e-001,-9.4013675e-001,-5.9718873e-001,-9.6613713e-001,-9.8445054e-001,-9.5208707e-001,-8.6506318e-001,-9.9784371e-001,-9.9603625e-001,-9.4019503e-001,-9.0484043e-001,-9.3381245e-001,-8.6938291e-001,-3.3967327e-001,-4.8580324e-001,-1.6625765e-001,-1.0000000e+000,-1.0000000e+000,-1.0000000e+000,1.1116947e-002,1.2125069e-001,-5.2294869e-001,-5.7199950e-001,-8.9461236e-001,-3.3826592e-001,-6.8679745e-001,1.8955250e-001,-1.1359571e-001,-9.9850627e-001,-9.9794262e-001,-9.9559788e-001,-9.9503559e-001,-9.9597690e-001,-9.9148743e-001,-9.9213116e-001,-9.9977512e-001,-9.9820395e-001,-9.9474673e-001,-9.9430903e-001,-9.9469323e-001,-9.9802046e-001,-9.9372747e-001,-9.9620895e-001,-9.9797615e-001,-9.9892091e-001,-9.9640106e-001,-9.9755992e-001,-9.9506803e-001,-9.9735126e-001,-9.9823805e-001,-9.9607053e-001,-9.9792459e-001,-9.9635664e-001,-9.9763980e-001,-9.9622004e-001,-9.9622539e-001,-9.2470550e-001,-9.9409728e-001,-9.9532347e-001,-9.9896944e-001,-9.9790162e-001,-9.9492962e-001,-9.8926072e-001,-9.8008540e-001,-9.3793669e-001,-9.9667757e-001,-9.9702540e-001,-9.8642738e-001,-9.3920079e-001,-9.9843184e-001,-8.9963316e-001,-9.3748500e-001,-9.2355140e-001,-9.2442913e-001,-9.4321038e-001,-9.4789152e-001,-8.9661455e-001,-9.3830911e-001,-9.4257570e-001,-9.4863426e-001,-9.5832542e-001,-9.5881686e-001,-9.4388241e-001,-9.8730333e-001,-9.7846559e-001,-9.0527425e-001,-9.9503609e-001,-9.9749931e-001,-9.9703070e-001,-8.8707736e-001,-9.3581956e-001,-9.5365327e-001,-4.7066160e-001,-6.7217180e-001,-5.9627404e-001,-5.2000000e-001,8.0000000e-002,3.2000000e-001,4.5100539e-001,1.3716703e-001,-1.8029913e-001,-5.8008614e-001,-9.0807003e-001,-6.2552686e-001,-9.4275716e-001,-6.6191006e-001,-9.1153211e-001,-9.9901148e-001,-9.9763006e-001,-9.9598477e-001,-9.9470846e-001,-9.9557158e-001,-9.8787651e-001,-9.8644523e-001,-9.9820517e-001,-9.9804041e-001,-9.9449682e-001,-9.9204962e-001,-9.8633303e-001,-9.9689353e-001,-9.9076269e-001,-9.9940916e-001,-9.9864806e-001,-9.9874875e-001,-9.9640627e-001,-9.9717222e-001,-9.9453020e-001,-9.9791979e-001,-9.9996970e-001,-9.9864167e-001,-9.9740170e-001,-9.9541085e-001,-9.9818240e-001,-9.9846306e-001,-9.9599581e-001,-9.9363885e-001,-9.9783780e-001,-9.9712931e-001,-9.9874950e-001,-9.9670876e-001,-9.9510966e-001,-9.9848581e-001,-9.9909354e-001,-9.9592463e-001,-9.9795094e-001,-9.9601290e-001,-9.9846008e-001,-9.9626705e-001,-9.9770046e-001,-8.2355788e-001,-8.0791598e-001,-9.1791256e-001,-9.0326274e-001,-8.2267700e-001,-9.5616508e-001,-8.6512704e-001,-8.3180082e-001,-9.4105617e-001,-9.0479801e-001,-8.7925663e-001,-9.6778849e-001,-8.7859934e-001,-9.4832915e-001,-9.1968517e-001,-8.2847203e-001,-9.9294953e-001,-9.8266311e-001,-9.9799327e-001,-8.7889440e-001,-8.3815197e-001,-9.2913995e-001,7.5814806e-004,2.0014368e-001,-2.5338416e-001,-1.0000000e+000,-9.3548387e-001,-9.3103448e-001,1.8403457e-001,-5.9322857e-002,4.3810716e-001,-3.9542276e-001,-6.9876160e-001,-3.8745724e-001,-7.8639417e-001,-4.8565359e-001,-7.8681512e-001,-9.9462908e-001,-9.9046914e-001,-9.9277782e-001,-9.9567969e-001,-9.8707772e-001,-9.8665097e-001,-9.8411445e-001,-9.8525021e-001,-9.9355255e-001,-9.9243025e-001,-9.8560582e-001,-9.8461702e-001,-9.9326921e-001,-9.9272545e-001,-9.7792782e-001,-9.9485223e-001,-9.9781915e-001,-9.9484295e-001,-9.9240991e-001,-9.8812267e-001,-9.9043323e-001,-9.8796127e-001,-9.8179096e-001,-9.9639919e-001,-9.9145229e-001,-9.8805999e-001,-9.8204835e-001,-9.9337999e-001,-9.9887769e-001,-9.9836665e-001,-9.9846730e-001,-9.9837462e-001,-9.9890591e-001,-9.9589361e-001,-9.9312836e-001,-9.9547231e-001,-9.9838710e-001,-9.9775430e-001,-9.9809593e-001,-9.9414765e-001,-9.9819058e-001,-9.9829005e-001,-7.9094643e-001,-7.1107400e-001,-7.2670699e-001,-7.7769715e-001,-9.4488134e-001,-7.9094643e-001,-9.5398356e-001,-8.7354261e-001,-1.7459288e-001,-1.0000000e+000,-4.8345254e-001,1.1040681e-002,-3.8451662e-001,-8.9506118e-001,-8.9635958e-001,-8.8819740e-001,-9.2846566e-001,-8.9809981e-001,-8.9506118e-001,-9.9347143e-001,-9.2147669e-001,-4.8461929e-001,-1.0000000e+000,-3.5355792e-002,-2.5424830e-001,-7.0032573e-001,-7.7061000e-001,-7.9711285e-001,-7.6448457e-001,-8.2018760e-001,-9.3795935e-001,-7.7061000e-001,-9.7095802e-001,-7.9838652e-001,1.7943523e-001,-1.0000000e+000,-4.7391298e-002,-4.6784901e-001,-7.6132577e-001,-8.9016545e-001,-9.0730756e-001,-8.9530057e-001,-9.1788296e-001,-9.0982876e-001,-8.9016545e-001,-9.9410543e-001,-8.9802151e-001,-2.3481529e-001,-1.0000000e+000,7.1645446e-002,-3.3037044e-001,-7.0597388e-001,6.4624029e-003,1.6291982e-001,-8.2588562e-001,2.7115145e-001,-7.2000927e-001,2.7680104e-001,-5.7978304e-002],0))