import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
import torch
# this method takes in all the moves in a game and encodes player, piece, file, rank, and modifiers
# `moves` is in the form {df}.loc[{game_index}, 'moves'].split()

def moves_vec(moves):

    move_vectors = np.zeros((100,5), dtype=int)
    
    player_idx = 0
    piece_idx = 1
    file_idx = 2
    rank_idx = 3
    mod_idx = 4

    piece_map = {'pawn': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    rank_list = np.arange(1,9)
    mod_map = {'O-O-O': 0, 'O-O': 1, 'x': 2, '+': 3, '#': 4, '=': 5} # queenside castle, kingside castle, capture, check, mate, promo
    
    for i, move in enumerate(moves):
        if(i%2==0):                            # encode PLAYER as index 0
            move_vectors[i][player_idx] = 0    # white move
        else:
            move_vectors[i][player_idx] = 1    # black move
            
            
        for key in piece_map:                  # encode PIECE as index 1 
            if key in move:
                move_vectors[i][piece_idx]=piece_map[key]
                
        if move_vectors[i][piece_idx] == 100:
            move_vectors[i][piece_idx] = 0    # set pawns
                
                
        for key in file_map:                   # encode FILES A-B as integers 1-8
            if key in move:
                move_vectors[i][file_idx] = file_map[key]
                
                
        for rank in rank_list:                 # encode RANK 0-7 as integers 1-8
            if str(rank) in move:
                move_vectors[i][rank_idx] = int(rank) - 1 

                
        for key in mod_map:                    # encode queenside castle, kingside castle, capture, check, mate, promo
            if key in move:
                move_vectors[i][mod_idx] = mod_map[key]

    return move_vectors       # outputs a (10, 5) vector, numpy array with dtypes np.int64

# this method takes in the winner of the game and returns a one hot vector 
# `winner` is in the form {df}.loc[{game_index}, 'winner']

def one_hot(winner):
    if(winner == 'white'):
        return [1, 0]     # white win, black loss
    else:
        return [0, 1]     # white loss, black win
    
class Normalize(object):                                                                       
    def __init__(self):
        
        self.piece_max = 6
        self.file_max = 7
        self.rank_max = 7
        self.mod_max = 5
                                                                                               
    def __call__(self,sample):                                                                 
        sample = sample.type(torch.FloatTensor)
        # index 0 - PLAYER
        
        # index 1 - PIECE  
        sample[:,1] = sample[:,1] / self.piece_max
              
        # index 2 - FILE
        sample[:,2] = sample[:,2] / self.file_max
        
        # index 3 - RANK
        sample[:,3] = sample[:,3] / self.rank_max
        
        # index 4 - MODS
        sample[:,4] = sample[:,4] / self.mod_max

        return sample
    
class ChessDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert data to PyTorch tensors
        #embed()
        moves = self.data.loc[idx, 'moves']
        moves_tensor = torch.tensor(moves_vec(moves))
        moves_tensor = self.transform(moves_tensor)
        
        winner = self.data.loc[idx, 'winner']
        label = torch.tensor(one_hot(winner))
        label = label.type(torch.FloatTensor)
        
        num_turns = self.data.loc[idx, 'turns']
        #print(f"index: {idx} num_turns: {num_turns}")
              
        return moves_tensor, label, num_turns

def get_dataset(filename):
    
    data = pickle.load(open(filename, "rb"))
    
    train_upper_bound = round(len(data)*0.8)

    train = data[:train_upper_bound]
    test = data[train_upper_bound:len(data)]
    test = test.reset_index()
    test = test.drop('index', axis = 1)
    
    normalize_transform = Normalize()
    train_data = ChessDataset(train, normalize_transform)
    test_data = ChessDataset(test, normalize_transform)
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
                         
    return train_loader, test_loader

# # Create train and test data loaders
# train, test = get_dataset()

# #pull a batch from dataloaders manually by setting batch_size=1 and running the following:
# batch = next(iter(train))
# print(len(batch[0][0]))
# batch2 = next(iter(test))
# print(batch2)