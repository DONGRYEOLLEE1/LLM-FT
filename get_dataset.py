from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Optional, List


def get_datasets(
    data_config: dict,
    shuffle: bool,
    columns: List[str],
    is_test: bool,
    test_size: Optional[float] = None
) -> DatasetDict:
    
    if type(data_config) is dict:
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")
    
    datasets = mix_datasets(
        dataset_mixer = dataset_mixer, 
        shuffle = shuffle, 
        columns = columns, 
        is_test = is_test, 
        test_size = test_size
    )
    
    return datasets

def mix_datasets(
    dataset_mixer: dict, 
    shuffle: bool = True, 
    columns: List[str] = ['question', 'chosen', 'rejected'], 
    changing_name_columns: Optional[List[str]] = None,
    changed_name_columns: Optional[List[str]] = None,
    is_test: bool = False, 
    test_size: Optional[float] = None
) -> DatasetDict:
    
    whole_dataset = DatasetDict()
    train_dataset = []
    test_dataset = []
    
    for name, sample_size in dataset_mixer.items():
        
        dataset = load_dataset(name, split = 'all')
        dataset = dataset.select(range(int(len(dataset) * sample_size)))
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns])
        
        if changing_name_columns is not None:
            
            if changed_name_columns is None:
                raise ValueError("changing_name_columns 값을 설정했으면 changed_name_columns도 설정해주세요.")
            if len(changed_name_columns) != len(changed_name_columns):
                raise ValueError("changing_name_columns와 changed_name_columns도 1대1 매칭하세요")
        
            
            
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


DATA_CONFIG = {
    "mncai/orca_dpo_pairs_ko": 0.7,
    "Ja-ck/Orca-DPO-Pairs-KO": 1
}

# data = get_datasets(data_config = DATA_CONFIG, shuffle = True, columns = ['question', 'chosen', 'rejected'], is_test = True, test_size = 0.01)

data = get_datasets(data_config = DATA_CONFIG, shuffle = True, columns = ['question', 'chosen', 'rejected'], is_test = False)