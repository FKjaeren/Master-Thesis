import torch
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, customer_id, article_id):
        self.customer_id = customer_id
        self. article_id = article_id
    def __len__(self):
        return len(self.customer_id)
    def __getitem__(self, transaction):
        customer_id = self.customer_id[transaction]
        article_id = self.article_id[transaction]
        #sample = {'customer_id':torch.tensor(customer_id, dtype = torch.long), 'article_id':torch.tensor(article_id, dtype = torch.int)}
        return torch.tensor(customer_id, dtype = torch.long), torch.tensor(article_id, dtype = torch.int)