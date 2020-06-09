import numpy as np

class M5Split2:
    '''
    Splits the data based on time keeping the train_size , val_size and step and gap_size maintained
    Example:
    |train + gap + val ---------------------------------------- |
    |step - train + gap + val --------------------------------- |
    |step step - train + gap + val ---------------------------- |

    '''
    
    def __init__(self, n_splits, group_id, date_col, train_size, gap_size, val_size, step):
        self.n_splits = n_splits + 1
        self.group_id = group_id
        self.date_col = date_col
        self.train_size = train_size
        self.gap_size = gap_size
        self.val_size = val_size
        self.step = step
        
        
    def split(self, df):
        df = df.sort_values(by=[self.date_col])
        indexes = []
        group_indexes = np.array(df.groupby(self.group_id, observed=True).apply(lambda x: x.index))
        for split in range(self.n_splits, 0, -1):
            val_idx = []
            gap_idx = []
            train_idx = []
                
            for idx_arr in group_indexes:
                
                if self.train_size + self.gap_size + self.val_size + self.step*split > len(idx_arr):
                    print(f'Max Split reached')
                    break
                    
                val_idx += list(idx_arr[-(self.val_size + self.step*(split-1)):len(idx_arr) - 1 - self.step*(split-1)])
                gap_idx += list(idx_arr[-(self.gap_size + self.val_size + self.step*(split-1)):-(self.val_size + self.step*(split-1))])
                train_idx += list(idx_arr[-(self.train_size + self.gap_size + self.val_size + self.step*(self.n_splits)):-(self.val_size + self.gap_size + self.step*(split-1))])
                
            yield train_idx, gap_idx, val_idx
