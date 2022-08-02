from torch import tensor
from obj_store import ObjStore

def test():
    store = ObjStore()
    status = store.save_objects(
        (
            "/var/foo/121.jpg",
            [("/var/foo/f/1", tensor([42,22,33,15]), 0.98, [], 22, "coco",
              "bulb"),
             ("/var/foo/f/2", tensor([96,44,36,26]), 0.97, [], 32, "coco",
              "vase")],
            [tensor([.2,.2,.3,.5]),
             tensor([.6,.0,.2,.1])],
        )
    )
    print(status)

    res = store.find_objects()
    for face in res:
        print(face)

    res = store.find_image_path("/var/foo/121.jpg")
    for im in res:
        print(im)

def test_file():
    store = ObjStore(sys.argv[1])
    res = store.find_objects()
    for face in res:
        print(face)

test()

