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

class ObjIter:
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

class ObjStore:
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
                        image_hash text
            );
            CREATE TABLE if not exists objects (
                        object_id integer,
                        image_id integer,
                        object_path text,
                        confidence real,
                        tag_id integer,
                        tag_type text,
                        tag text,
                        object_box pt_tensor, 
                        embedding pt_tensor
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

    def save_objects(self, image_path,fs,es):
        """
        (path, object_desciptors[n], embeddings[n])
        where 
            object_descriptor: (objfile, box, confidence, [], tag_id, tag_type, tag)
        """
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        idr, bna = self.split_path(image_path)
        image_id = self.next_idx()
        image = (image_id, now, idr, bna, "hash")
        self.cur.execute("insert into images values (?, ?, ?, ?, ?)", image)
        for fi in list(fs):
            obj_id = self.next_idx()
            obj = (obj_id, image_id, fi[0], float(fi[2])
                  , fi[4], fi[5], fi[6]
                  , tensor(fi[1])
                  , tensor([])
                  )
            # print(">>", obj)
            self.cur.execute("insert into objects values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             obj)
        self.con.commit()
        return 0

    def find_objects(self):
        self.cur.execute('select * from objects')
        return ObjIter(self.cur)
    
    def find_image_path(self, image_path):
        self.cur.execute("select created from images where image_dir = '%s' and image_name = '%s'"
                        % self.split_path(image_path))
        result = self.cur.fetchall()
        return result

