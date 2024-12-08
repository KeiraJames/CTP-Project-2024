# imports
import base64
import os
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

# load model weights
@st.cache_resource
def load_models():
    # Load all models at once
    eff_net_model = tf.keras.models.load_model('EfficientNet_Models/efficientnetb3_binary_classifier_8.h5')
    eff_net_art_model = tf.keras.models.load_model('EfficientNet_Models/EfficientNet_fine_tune_art_model.h5')
    cnn_model = 'CNN_model_weight/model_weights.weights.h5'
    return eff_net_model, eff_net_art_model, cnn_model

# Access cached models
eff_net_model, eff_net_art_model, cnn_model = load_models()

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

# efficientnet model
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
 
# efficientnet art model
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

# preprocess images for efficient net
def pre_process_img_effNet(image):
    img = load_img(image, target_size=(300, 300))  # Resize image to model input size
    img_arr = img_to_array(img)  # Convert to array
    img_arr = np.expand_dims(img_arr, axis=0) # Add batch dimension
    result = run_effNet(img_arr)
    return result
# preprocess images for efficient net art
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


#UI

#title
col1, col2, col3,col4, col5 = st.columns([4,1,3,3,1],  gap="small")

# In the first column, display the image
with col1:
    st.write('')
with col2:
    st.image("styles/robot.png")

# In the second column, display the text
with col3:
    st.markdown(
        """
        <p class="title"> AI vs REAL Image Detection </p>
        """,
        unsafe_allow_html=True
    )
with col4:
    st.write('')
with col5:
    st.write('')

# division between photo and other widget component
main_col_one, main_col_two = st.columns([2,2], gap="large")
#photo column
with main_col_one:
    # Create a placeholder for the image
    image_placeholder = st.empty()

with main_col_two:
    with open("styles/detectiveMag.svg", "r") as file:
        svg_content_detective_Mag = file.read()

    #alignment between magnifying glass image and upload line
    col1, col2, col3,col4 = st.columns([4,4,1,3],  gap="small")
    with col1:
        st.write('')
    with col2:
        st.markdown(
            """<p class = "upload_line"> Please upload the image </p>""",
            unsafe_allow_html= True
        )
    with col3:
        st.markdown(
        f"<div class='detectiveMag1' style='bottom: 0%; right: 0%;'>{svg_content_detective_Mag}</div>",
        unsafe_allow_html=True
    )
    with col4:
        st.write('')


                
    # introduce states
    if "prev_image" not in st.session_state:
        st.session_state.prev_image = None 
    if "reset_model" not in st.session_state:
        st.session_state.reset_model = False
    if "model_key" not in st.session_state:
        st.session_state.model_key = "default_model_key"



    # Upload image widget
    user_image = st.file_uploader("png, jpg, or jpeg image", ['png', 'jpg', 'jpeg'], label_visibility='hidden')

    if user_image:
    # Convert the image to base64 encoding
        image_bytes = user_image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Display the image centered using HTML
        image_placeholder.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="data:image/jpeg;base64,{image_base64}" max-width:"100%" height:"auto"/>'
            f'</div>',
            unsafe_allow_html=True
        )

    # model name select box widget reset condition. reset model name when a new image is uploaded
    if user_image != st.session_state.prev_image:
        if st.session_state.prev_image is not None: 
            st.session_state.model_key = "reset_model_key" if st.session_state.model_key == "default_model_key" else "default_model_key"
            st.session_state.reset_model = True
        st.session_state.prev_image = user_image  # set prev image to current image 

    # model name select box widget
    model_name = st.selectbox(
        'Choose a model',
        ['CNN', 'Efficientnet', 'Efficientnet Art'],
        index=None,
        placeholder='choose an option',
        key=st.session_state.model_key
    )

    # placeholder to display result
    result_placeholder = st.empty()

if user_image is not None and model_name is not None:
    predictions = []
    # preprocess image and run the user selected model
    if model_name == 'CNN':
        print('CNN is running')
        predictions = pre_process_img(user_image)
    elif model_name == 'Efficientnet':
        print('Effnet is running')
        predictions = pre_process_img_effNet(user_image)
    elif model_name == 'Efficientnet Art':
        print('Effnet Art is running')
        predictions = pre_process_img_effNetArt(user_image)

    if predictions[0] < 0.5:
         result_word = "AI Generated"
    else:
         result_word = "REAL"

    # display the result and the prediction
    if user_image is not None:
        if len(predictions) > 0: 
            result_placeholder.markdown(f"<div class='result'> <span class = 'prediction'>Prediction: {predictions[0][0]:.2%}</span> <br> It is a <span class = resultword> {result_word} </span> image. </div>", unsafe_allow_html=True)

    print(model_name)
    print(predictions[0])



