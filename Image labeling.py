import os

directory = "D:\\DRDO (Dare to dream 4.0 competition)\\Electronic component data set\\images\\armature"
new_prefix = "armature_"

# get a list of all image files in the directory
image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# rename each file in the list with the new prefix and a sequential number
for i, old_name in enumerate(image_files):
    new_name = new_prefix + str(i+1) + os.path.splitext(old_name)[1] # get the file extension and add it to the new name
    os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))

