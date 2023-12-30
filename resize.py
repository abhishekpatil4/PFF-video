from PIL import Image

# Specify the input and output image paths
input_image_path = "UCF101_subset/train/ApplyEyeMakeup/frame_0000.png"
output_image_path = "./resized-img.png"

# Open the input image
image = Image.open(input_image_path)

# Resize the image to 32x32
resized_image = image.resize((32, 32))

# Save the resized image
resized_image.save(output_image_path)
