from face_utils import get_image_files

for f in get_image_files("data/objects"):
    print(f)

for f in get_image_files("/media/Windows/__root/2022-07-11"):
    print(f)
