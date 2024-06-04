from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Optional, List


class DatasetLoader:
    def __init__(self, data_config: dict, columns: List[str] = ['instruction', 'input', 'output'], changing_columns: Optional[List[str]] = None, changed_columns: Optional[List[str]] = None):
        
        self.data_config = data_config
        self.columns = columns
        self.changing_columns = changing_columns
        self.changed_columns = changed_columns
        
    def get_datasets(self, test_size: Optional[int], shuffle: bool = True, is_test: bool = False) -> DatasetDict:
        
        if type(self.data_config) is not dict:
            raise ValueError(f"Data config {self.data_config} not recognized.")
        
        datasets = self._mix_datasets(shuffle = shuffle, is_test = is_test, test_size = test_size)
        
        return datasets
    
    def _mix_datasets(self, test_size: Optional[int], shuffle: bool = True, is_test: bool = False) -> DatasetDict:
        
        whole_dataset = DatasetDict()
        train_dataset = []
        test_dataset = []
        
        for name, sample_size in self.data_config.items():
            dataset = load_dataset(name, split = "all")
            dataset = dataset.select(range(int(len(dataset) * sample_size)))
            if self.changing_columns is not None:
                if not isinstance(self.changing_columns, list) or not isinstance(self.changed_columns, list):
                    raise ValueError("changing_columns, changed_name_columns은 반드시 List 형태로 들어와야합니다.")
                if len(self.changing_columns) != len(self.changed_columns):
                    raise ValueError("changing_columns와 changed_columns은 sequential 해야하며 1대1 매칭되어야 합니다.")
                
                if any(item not in dataset.column_names for item in self.changing_columns):
                    pass
                else:
                    new_column_names = {old: new for old, new in zip(self.changing_columns, self.changed_columns)}
                    dataset = dataset.rename_columns(new_column_names)
                
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in self.columns])
            
            if shuffle:
                dataset = dataset.shuffle(seed = 42)
                
            if is_test:
                dataset = dataset.train_test_split(test_size = test_size)
                test_dataset.append(dataset['test'])
                train_dataset.append(dataset['train'])
            else:
                dataset = DatasetDict({"train": dataset})
                train_dataset.append(dataset['train'])
                
        if len(train_dataset) > 0:
            whole_dataset['train'] = concatenate_datasets([li_train for li_train in train_dataset])
            
        if len(test_dataset) > 0:
            whole_dataset['test'] = concatenate_datasets([li_test for li_test in test_dataset])
            
        return whole_dataset
                
    def data2json(self, file_path: str) -> None:
        dataset_dict = {}
        dataset_source = {}
        for name, sample_size in self.data_config.items():
            dataset = load_dataset(name, split = "all")
            dataset = dataset.select(range(int(len(dataset) * sample_size)))
            
            dataset_length = len(dataset)
            
            dataset_source[name] = dataset_length
        
        dataset_dict['dataset'] = dataset_source
        
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f, indent = 4, ensure_ascii = False)