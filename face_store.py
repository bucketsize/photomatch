import sqlite3
from datetime import datetime as dt

class Box:
    def __init__(self, box):
        self.box = box

    def __repr__(self):
        x1,y1,x2,y2 = self.box
        return "Box[%d,%d,%d,%d]" % (x1,y1,x2,y2)

def box_adapter(box):
    x1,y1,x2,y2 = box.box
    return ("%d;%d;%d;%d" % (x1,y1,x2,y2)).encode('ascii')

def box_converter(s):
    box = list(map(int, s.split(b";")))
    return box

sqlite3.register_adapter(Box, box_adapter)
sqlite3.register_converter("box", box_converter)

class FaceIter:
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

class FaceStore:
    def __init__(self):
        con = sqlite3.connect('.faces.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.execute('''CREATE TABLE if not exists faces
               (date text, image_path text, face_id text, face_box box)''')
        con.commit()
        self.store = (con, cur)

    def save_faces(self, image_path, faces):
        con, cur = self.store
        cur.execute("select date from faces where image_path = '%s'" % image_path)
        result = cur.fetchall()
        if len(result) > 0:
            return 1
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        def face_ent(face):
            box = Box(face["box"])
            face_id = image_path + "_" + str(box)
            return (now, image_path, face_id, box)
        face_list = list(map(face_ent, faces))
        print(face_list)
        cur.executemany("insert into faces values (?, ?, ?, ?)", face_list)
        con.commit()
        return 0

    def find_faces(self):
        con, cur = self.store
        cur.execute('select * from faces')
        return FaceIter(cur)

    def find_image_path(self, image_path):
        con, cur = self.store
        cur.execute("select date from faces where image_path = '%s'" % image_path)
        result = cur.fetchall()
        return result

def test():
    store = FaceStore()
    status = store.save_faces("/var/foo/121.jpg", [[42,22,33,15], [56,40,32,21]])
    print(status)

    res = store.find_faces()
    for face in res:
        print(face)

    res = store.find_image_path("/var/foo/121.jpg")
    for im in res:
        print(im)

# test()
