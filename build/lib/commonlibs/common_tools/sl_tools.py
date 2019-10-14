import json
import pickle as pkl
import traceback

def pklsave(obj, file_path):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        print('SAVE OBJ: %s' % file_path)

def jsonsave(obj, file_path):
    with open(file_path, 'wt+') as f:
        json.dump(obj, f)
        print('SAVE JSON: %s' % file_path)

def pklload(file_path):
    with open(file_path, 'rb') as f:
        print('LOAD OBJ: %s' % file_path)
        try:
            return pkl.load(f)
        except Exception:
            traceback.print_exc()

def jsonload(file_path):
    with open(file_path, 'r') as f:
        print('LOAD OBJ: %s' % file_path)
        try:
            return json.load(f)
        except EOFError:
            print('EOF Error %s' % file_path)




