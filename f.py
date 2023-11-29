
    import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from os import listdir, mkdir
from os.path import isfile, join, splitext, exists

def process_image(file_path, n_colors, resize_percent):
    # Load and preprocess the image using VGG16 preprocessing
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = VGG16.preprocess_input(img_array)

    # Create a pre-trained VGG16 model without the top (classification) layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Get feature maps from the convolutional layers
    features = base_model.predict(img_array)


    flattened_features = features.reshape(-1, features.shape[-1])
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(flattened_features)
    labels = kmeans.predict(flattened_features)
    clusters = kmeans.cluster_centers_


    result = clusters[labels].reshape(features.shape)

    plt.imshow(result[0]/255)
    plt.show()

    img = Image.fromarray((result[0] * 255).astype(np.uint8))
    if not exists(file_path + "/Reduced"):
        mkdir(file_path + "/Reduced")

    new_file_name = join(file_path + "/Reduced", splitext(file)[0] + "Reduced" + ".jpg")

    if resize_percent < 100:
        new_w = int((resize_percent / 100) * img.size[0])
        new_h = int((resize_percent / 100) * img.size[1])
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img.save(new_file_name)

directory_name = input("Directory name ? ")
n_colors = int(input("Number of colors ? "))
resize_percent = float(input("Percent resize ? "))


only_files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
only_files = [f for f in only_files if str(f).endswith("jpg")]

for file in only_files:
    print(file)
    source_file_name = join(directory_name, file)
    process_image(source_file_name, n_colors, resize_percent)
