from abc import *
from pathlib import Path
import pickle


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, user2dict, user_count, item_count, sample_size, seed, save_folder):
        # 유저별 아이템 index list(type : zip)
        self.user2dict = user2dict
        # 유저 수(len(umap))
        self.user_count = user_count
        # 아이템 갯수(len(smap))
        self.item_count = item_count
        # templates에서 설정한 옵션
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    
    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-seed{}.pkl'.format(self.code(), self.sample_size, self.seed)
        return folder.joinpath(filename)
