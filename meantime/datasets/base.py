from .utils import *

from tqdm import tqdm
from dotmap import DotMap

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle
from datetime import date

tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.local_data_folder = args.local_data_folder

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    # 무슨 데이터인지 ex)ml-1m
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    # data load
    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    # 데이터 전처리
    def preprocess(self):
        # 데이터 저장 경로
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return

        # 데이터가 경로에 없으면, 폴더 생성
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        # 데이터 다운
        self.maybe_download_raw_dataset()
        # open dataset
        df = self.load_ratings_df()
        # 데이터의 rating이 기준 이하인 경우 버림
        df = self.make_implicit(df)
        # 일정 갯수 이하인 데이터 버림(user, item)
        df = self.filter_triplets(df)

        # umap, smap : 유저, 아이템 라벨 인코딩
        # df : 라벨인코딩 된 데이터
        df, umap, smap = self.densify_index(df)

        # 
        user2dict, train_targets, validation_targets, test_targets = self.split_df(df, len(umap))

        special_tokens = DotMap()
        special_tokens.pad = 0
        item_count = len(smap)
        special_tokens.mask = item_count + 1
        special_tokens.sos = item_count + 2
        special_tokens.eos = item_count + 3

        num_ratings = len(df)
        num_days = df.days.max() + 1

        dataset = {'user2dict': user2dict,
                   'train_targets': train_targets,
                   'validation_targets': validation_targets,
                   'test_targets': test_targets,
                   'umap': umap,
                   'smap': smap,
                   'special_tokens': special_tokens,
                   'num_ratings': num_ratings,
                   'num_days': num_days}

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    # rating이 기준 이하일 경우 버림
    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    # 
    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            # 아이템 id별 갯수
            item_sizes = df.groupby('sid').size()
            # 아이템 갯수가 기준값보다 이상인 것을 index
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            # 위의 index를 기준으로 아이템 filtering
            df = df[df['sid'].isin(good_items)]

        # 위와 같은 기준을 user에 적용
        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    # user-item matrix 생성
    def densify_index(self, df):
        print('Densifying index')
        # user, item 라벨 인코딩(중복값 제거)
        umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}
        # user-item matrix 생성
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        # timestamp순으로 정렬 하는 함수
        def sort_by_time(d):
            d = d.sort_values(by='timestamp')
            return {'items': list(d.sid), 'timestamps': list(d.timestamp), 'ratings': list(d.rating), 'days': list(d.days)}

        # 최초 구매기록
        min_date = date.fromtimestamp(df.timestamp.min())
        # 데이터에서 최초 구매일자 대비 며칠이 지났는지
        df['days'] = df.timestamp.map(lambda t: (date.fromtimestamp(t) - min_date).days)
        user_group = df.groupby('uid')
        # user2dict : uid별 구매 item
        user2dict = user_group.progress_apply(sort_by_time)
        
        if self.args.split == 'leave_one_out':
            train_ranges = []
            val_positions = []
            test_positions = []
            for user, d in user2dict.items():
                n = len(d['items'])
                train_ranges.append((user, n-2))  # exclusive range
                val_positions.append((user, n-2))
                test_positions.append((user, n-1))
            train_targets = train_ranges
            validation_targets = val_positions
            test_targets = test_positions
        else:
            raise ValueError

        return user2dict, train_targets, validation_targets, test_targets

    def _get_rawdata_root_path(self):
        return Path(self.local_data_folder)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

