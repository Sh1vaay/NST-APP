import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model 
model = tf.saved_model.load(r'./saved/')

def main():
    st.title('Neural Style Transfer')

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox('Menu',menu) 

    if choice == 'Home':
        st.subheader('Home')
        st.write('Welcome to the Home page!')

        st.write('Please upload two images. The first will be used as the content image, and the second will be used as the style image.')

        content_file = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'])
        style_file = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'])

        if content_file is not None and style_file is not None:
            content_image = Image.open(content_file)
            style_image = Image.open(style_file)

            st.image(content_image, caption='Content Image', use_column_width=True)
            st.image(style_image, caption='Style Image', use_column_width=True)

            if st.button('Convert'):
                # Change the dimensions to your desired size
                desired_size = (1024, 1024)

                content_image = tf.image.resize(np.array(content_image), desired_size)
                style_image = tf.image.resize(np.array(style_image), desired_size)
                
                # Preprocess the images
                content_image_np = content_image.numpy().astype(np.float32)[np.newaxis, ...] / 255.
                style_image_np = style_image.numpy().astype(np.float32)[np.newaxis, ...] / 255.

                # Ensure the images are the same size
                content_image_np = tf.image.resize(content_image_np, (256, 256))
                style_image_np = tf.image.resize(style_image_np, (256, 256))

                # Apply style transfer
                outputs = model(tf.constant(content_image_np), tf.constant(style_image_np))
                stylized_image = outputs[0]

                # Adjust saturation
                stylized_image = tf.image.adjust_saturation(stylized_image, 2)

                # Convert the output tensor to a numpy array
                stylized_image_np = stylized_image.numpy()

                # Ensure the image is in the correct format for displaying
                stylized_image_np = np.squeeze(stylized_image_np)
                stylized_image_np = np.clip(stylized_image_np * 255, 0, 255).astype('uint8')

                # Convert the numpy array to a PIL Image and then display it
                stylized_image_pil = Image.fromarray(stylized_image_np)
                st.image(stylized_image_pil, caption='Stylized Image', width=400)

    elif choice == 'About':
        st.subheader('About')
        st.write('This is a simple Streamlit app.')

if __name__ == "__main__":
    main()
