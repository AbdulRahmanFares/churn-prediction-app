import pickle

def save_pickle(filepath, obj):
    with open(filepath, mode='wb') as file:
        pickle.dump(obj, file)

def load_pickle(filepath):
    with open(filepath, mode='rb') as file:
        obj = pickle.load(file)
    
    return obj
