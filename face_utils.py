import glob
import ssl
import urllib.request
import pprint

pp = pprint.PrettyPrinter(width=41, compact=True)
def print_nice(o):
    pp.pprint(o)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def http_get_file(url, local_file_name):
    if os.path.exists(local_file_name):
        print("cachehit: %s", local_file_name)
        return 
    else:
        with urllib.request.urlopen(url, context=ctx) as resource:
            with open(local_file_name, 'wb') as f:
                f.write(resource.read())

def get_image_files(base_dir):
    return glob.iglob(base_dir+"/**/*.jpg", recursive=True)

def in_range(r, i):
    (x1, x2) = r
    return (x1<=i and i<=x2) 

def in_box(r, x, y):
    (x1, y1, x2, y2) = r
    return (x1<=x and x<=x2 and y1<=y and y<=y2)


