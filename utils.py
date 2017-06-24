import numpy as np
import json
def load_imdb_data(imdb_path="imdb.npz", 
              word_index_json_path="imdb_word_index.json", 
              seed=0):
    
    with open(word_index_json_path) as f_json:
        wid_dict = json.load(f_json)
    
    id2w = {}
    for (k, v) in wid_dict.items():
        id2w[v] = k
    with np.load(imdb_path) as dataset:
        x_train = dataset['x_train']
        x_test = dataset['x_test']
        y_train = dataset['y_train']
        y_test  = dataset['y_test']
        
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    
    np.random.seed(seed)
    np.random.shuffle(x_test)
    np.random.seed(seed)
    np.random.shuffle(y_test)
    
    xs = np.concatenate([x_train, x_test])
    ys = np.concatenate([y_train, y_test])
    
    # pad_char == 0 start_char == 1 oov == 2
    start_from = 3
    xs = [[w + start_from for w in x] for x in xs]
    xs_length = [len(s) for s in xs]
    idx = len(x_train)
    
    x_train, y_train = np.array(xs[:idx]), np.array(ys[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(ys[idx:])
    train_length, test_length = np.array(xs_length[:idx]), np.array(xs_length[idx:])

    return x_train, y_train, x_test, y_test, train_length, test_length, wid_dict, id2w