import os
import io
import sys
import time
import torch
from torch import tensor
import sqlite3
from datetime import datetime as dt
from functools import reduce

def pt_tensor_adapter(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

def pt_tensor_converter(bs):
    buf = io.BytesIO(bs)
    return torch.load(buf) 

sqlite3.register_adapter(torch.Tensor, pt_tensor_adapter)
sqlite3.register_converter("pt_tensor", pt_tensor_converter)

class ImgIter:
    def __init__(self, cursor):
        self.count = 1
        self.cursor = cursor

    def __iter__(self):
        return self

    def __next__(self):
        r = self.cursor.fetchone()
        if r is None:
            raise StopIteration
        self.count += 1
        return r 

class ImgStore:
    def __init__(self, db_file=":memory:"):
        print("using db", db_file)
        con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.executescript('''
            CREATE TABLE if not exists images (
                        image_id integer,
                        created text,
                        image_dir text,
                        image_name text,
                        image_hash text,
                        imagenet_id integer,
                        imagenet_cl text
            );
        ''')
        con.commit()
        self.con = con
        self.cur = cur
        self.index = int(time.time())

    def next_idx(self):
        self.index += 1
        return self.index

    def split_path(self, image_path):
        bna = os.path.basename(image_path)
        idr = image_path[0:(len(image_path)-len(bna))]
        return (idr, bna)

    def save_image(self, image):
        image_path, imagenet_id, imagenet_cl = image
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        idr, bna = self.split_path(image_path)
        image_id = self.next_idx()
        image = (image_id, now, idr, bna, "hash", imagenet_id, imagenet_cl)
        self.cur.execute("insert into images values (?, ?, ?, ?, ?, ?, ?)", image)
        self.con.commit()
        return 0

    def find_images(self):
        self.cur.execute('select * from images')
        return ImgIter(self.cur)
    
    def find_image_path(self, image_path):
        self.cur.execute("select created from images where image_dir = '%s' and image_name = '%s'"
                        % self.split_path(image_path))
        result = self.cur.fetchall()
        return result

def test():
    store = ImgStore()
    status = store.save_image(
        (
            "/var/foo/121.jpg", 100, "boat, dingy"
        )
    )
    print("status: ", status)

    res = store.find_images()
    for face in res:
        print("image: ", face)

    res = store.find_image_path("/var/foo/121.jpg")
    for im in res:
        print("image_path: ", im)

def test_file():
    store = ImgStore(sys.argv[1])
    res = store.find_images()
    for face in res:
        print(face)

# test()
# test_file()
