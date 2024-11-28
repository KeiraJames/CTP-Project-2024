# imports
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np





# load css
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("./styles/style.css")


# bootstrap
st.markdown(
    """<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">""",
    unsafe_allow_html=True
)


eff_net_model = tf.keras.models.load_model('EfficientNet_Models/efficientnetb3_binary_classifier_8.h5')
eff_net_art_model = tf.keras.models.load_model('EfficientNet_Models/EfficientNet_fine_tune_art_model.h5')
cnn_model = 'CNN_model_weight/model_weights.weights.h5'

# CNN model
def run_cnn(img_arr):
    my_model = Sequential()
    my_model.add(Conv2D(
            filters=16, 
            kernel_size=(3, 3), 
            strides=(1, 1),
            activation='relu',
            input_shape=(256, 256, 3) 
    ))
    my_model.add(BatchNormalization())
    my_model.add(MaxPooling2D())
    
    my_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')) 
    my_model.add(BatchNormalization())
    my_model.add(MaxPooling2D()) 

    my_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) 
    my_model.add(BatchNormalization())
    my_model.add(MaxPooling2D())
    
    my_model.add(Flatten())
    my_model.add(Dense(512, activation='relu')) 
    my_model.add(Dropout(0.09)) 
    my_model.add(Dense(1, activation='sigmoid'))
    my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Load the pre-trained weights
    my_model.load_weights(cnn_model)

    prediction = my_model.predict(img_arr)
    return prediction

def run_effNet(img_arr):
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        strategy = tf.distribute.get_strategy()  
    with strategy.scope():
            prediction = eff_net_model.predict(img_arr)
    return prediction
 
   
def run_effNet_Art(img_arr):
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        strategy = tf.distribute.get_strategy()  
    with strategy.scope():
            prediction = eff_net_art_model.predict(img_arr)
    return prediction

def pre_process_img_effNet(image):
    img = load_img(image, target_size=(300, 300))  # Resize image to model input size
    img_arr = img_to_array(img)  # Convert to array
    img_arr = np.expand_dims(img_arr, axis=0) # Add batch dimension
    result = run_effNet(img_arr)
    return result

def pre_process_img_effNetArt(image):
    img = load_img(image, target_size=(224, 224))  # Resize image to model input size
    img_arr = img_to_array(img)  # Convert to array
    img_arr = np.expand_dims(img_arr, axis=0) # Add batch dimension
    result = run_effNet_Art(img_arr)
    return result

# preprocess image for cnn
def pre_process_img(image):
        # Load and preprocess the image
        input_picture = load_img(image, target_size=(256, 256))
        img_arr = img_to_array(input_picture) / 255.0  # Normalize the image
        img_arr = img_arr.reshape((1, 256, 256, 3))  # Add batch dimension
        result = run_cnn(img_arr)
        return result
# title
st.markdown(
    """<p class = "title"> AI vs REAL Image Detection </p>""",
    unsafe_allow_html= True
)

# upload image
st.markdown(
     """<p class = "upload_line"> Please upload the image </p>""",
    unsafe_allow_html= True
)
user_image = st.file_uploader("png, jpg, or jpeg image", ['png', 'jpg', 'jpeg'], label_visibility='hidden')
model_name = st.selectbox('Choose a model', ['CNN', 'Efficiencynet', 'Efficiencynet Art'], index=None, placeholder='choose an option')
result_placeholder = st.empty()

# design animation elements
with open("styles/detectiveMag.svg", "r") as file:
    svg_content_detective_Mag = file.read()

# First magnifying glass starts at bottom-right
st.markdown(
    f"<div class='detectiveMag1' style='bottom: 0%; right: 0%;'>{svg_content_detective_Mag}</div>",
    unsafe_allow_html=True
)

# Second magnifying glass starts slightly higher up the diagonal
st.markdown(
    f"<div class='detectiveMag2' style='bottom: 10%; right: 10%;'>{svg_content_detective_Mag}</div>",
    unsafe_allow_html=True
)

# Third magnifying glass starts further up the diagonal
st.markdown(
    f"<div class='detectiveMag3' style='bottom: 20%; right: 20%;'>{svg_content_detective_Mag}</div>",
    unsafe_allow_html=True
)

if user_image is not None and model_name is not None:
    predictions = []
    if model_name == 'CNN':
        print('CNN is running')
        predictions = pre_process_img(user_image)
    elif model_name == 'Efficiencynet':
        print('Effnet is running')
        predictions = pre_process_img_effNet(user_image)
    elif model_name == 'Efficiencynet Art':
        print('Effnet Art is running')
        predictions = pre_process_img_effNetArt(user_image)

    if predictions[0] < 0.5:
         result_word = "FAKE"
    else:
         result_word = "REAL"

    if user_image is not None:
        if len(predictions) > 0: 
            result_placeholder.markdown(f"<div class='result'> It is a <span class = resultword> {result_word} </span> image </div>", unsafe_allow_html=True)

    print(model_name)
    print(predictions[0])



