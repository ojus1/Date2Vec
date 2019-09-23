from torch.utils.data import Dataset
import numpy as np

class NextDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        return np.array(self.dates[idx]).astype(np.float32), np.array(self.dates[idx+1]).astype(np.float32)

class TimeDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        x = np.array(self.dates[idx]).astype(np.float32)
        return x, x

if __name__ == "__main__":
    dt = open("dates.txt", 'r').readlines()
    dataset = NextDateDataset(dt)
    print(dataset[0])