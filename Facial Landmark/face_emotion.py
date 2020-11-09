from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tqdm import tqdm

def loadModel():
    num_classes = 7
    model = Sequential()

	#1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    model.add(Flatten())

	#fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))
	
	#----------------------------
    model.load_weights(filepath='facial_expression_model_weights.h5')
    return model

def analyze(img_path, actions = [], models = {}, enforce_detection = True, detector_backend = 'opencv'):

	if type(img_path) == list:
		img_paths = img_path.copy()
		bulkProcess = True
	else:
		img_paths = [img_path]
		bulkProcess = False

	#---------------------------------

	#if a specific target is not passed, then find them all
	if len(actions) == 0:
		actions= ['emotion', 'age', 'gender', 'race']

	#print("Actions to do: ", actions)

	#---------------------------------

	if 'emotion' in actions:
		if 'emotion' in models:
			print("already built emotion model is passed")
			emotion_model = models['emotion']
		else:
			emotion_model = Emotion_loadModel()

	if 'age' in actions:
		if 'age' in models:
			print("already built age model is passed")
			age_model = models['age']
		else:
			age_model = Age.loadModel()

	if 'gender' in actions:
		if 'gender' in models:
			print("already built gender model is passed")
			gender_model = models['gender']
		else:
			gender_model = Gender.loadModel()

	if 'race' in actions:
		if 'race' in models:
			print("already built race model is passed")
			race_model = models['race']
		else:
			race_model = Race.loadModel()
	#---------------------------------

	resp_objects = []
	
	disable_option = False if len(img_paths) > 1 else True
	
	global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing', disable = disable_option)
	
	#for img_path in img_paths:
	for j in global_pbar:
		img_path = img_paths[j]

		resp_obj = "{"
		
		disable_option = False if len(actions) > 1 else True

		pbar = tqdm(range(0,len(actions)), desc='Finding actions', disable = disable_option)

		action_idx = 0
		img_224 = None # Set to prevent re-detection
		#for action in actions:
		for index in pbar:
			action = actions[index]
			pbar.set_description("Action: %s" % (action))

			if action_idx > 0:
				resp_obj += ", "

			if action == 'emotion':
				emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
				img = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend)

				emotion_predictions = emotion_model.predict(img)[0,:]

				sum_of_predictions = emotion_predictions.sum()

				emotion_obj = "\"emotion\": {"
				for i in range(0, len(emotion_labels)):
					emotion_label = emotion_labels[i]
					emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

					if i > 0: emotion_obj += ", "

					emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)

				emotion_obj += "}"

				emotion_obj += ", \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])

				resp_obj += emotion_obj

			elif action == 'age':
				if img_224 is None:
					img_224 = functions.preprocess_face(img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection) #just emotion model expects grayscale images
				#print("age prediction")
				age_predictions = age_model.predict(img_224)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)

				resp_obj += "\"age\": %s" % (apparent_age)

			elif action == 'gender':
				if img_224 is None:
					img_224 = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend) #just emotion model expects grayscale images
				#print("gender prediction")

				gender_prediction = gender_model.predict(img_224)[0,:]

				if np.argmax(gender_prediction) == 0:
					gender = "Woman"
				elif np.argmax(gender_prediction) == 1:
					gender = "Man"

				resp_obj += "\"gender\": \"%s\"" % (gender)

			elif action == 'race':
				if img_224 is None:
					img_224 = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend) #just emotion model expects grayscale images
				race_predictions = race_model.predict(img_224)[0,:]
				race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

				sum_of_predictions = race_predictions.sum()

				race_obj = "\"race\": {"
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions

					if i > 0: race_obj += ", "

					race_obj += "\"%s\": %s" % (race_label, race_prediction)

				race_obj += "}"
				race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])

				resp_obj += race_obj

			action_idx = action_idx + 1

		resp_obj += "}"

		resp_obj = json.loads(resp_obj)

		if bulkProcess == True:
			resp_objects.append(resp_obj)
		else:
			return resp_obj

	if bulkProcess == True:
		resp_obj = "{"

		for i in range(0, len(resp_objects)):
			resp_item = json.dumps(resp_objects[i])

			if i > 0:
				resp_obj += ", "

			resp_obj += "\"instance_"+str(i+1)+"\": "+resp_item
		resp_obj += "}"
		resp_obj = json.loads(resp_obj)
		return resp_obj
		#return resp_objects