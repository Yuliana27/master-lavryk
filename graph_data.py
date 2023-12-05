from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)

class BeautyGraph(InMemoryDataset):
    def __init__(self, beauties_unique, beauty_data, transform=None):
        super(BeautyGraph, self).__init__('.', transform, None, None)

        data = HeteroData()

        #nodes-beauty
        beauty_mapping = {idx: i for i, idx in enumerate(beauties_unique['product_id_encoded'])}

        beauty_categories = beauties_unique['class'].str.get_dummies().values
        beauty_categories = torch.from_numpy(beauty_categories).to(torch.float)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        with torch.no_grad():
            emb = model.encode(beauties_unique['asin'].values, show_progress_bar=True, convert_to_tensor=True).cpu()
            print(emb)
            print(len(emb[0]))
            print(len(emb))

        data['beauties'].x = torch.cat([emb, beauty_categories], dim=-1)
        data['beauties'].num_nodes = len(beauty_mapping)

        #nodes-user
        user_mapping = {idx: i for i, idx in enumerate(beauty_data['reviewer_id'].unique())}
        data['user'].num_nodes = len(user_mapping)
        data['user'].x = torch.eye(data['user'].num_nodes)

        #edges
        src = [user_mapping[idx] for idx in beauty_data['reviewer_id']]
        dst = [beauty_mapping[idx] for idx in beauty_data['product_id_encoded']]
        edge_index = torch.tensor([src, dst])


        rating = torch.from_numpy(beauty_data['rating'].values).to(torch.long)
        data['user', 'rates', 'beauties'].edge_index = edge_index
        data['user', 'rates', 'beauties'].edge_label = rating

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)