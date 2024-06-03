from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Optional, List


def get_datasets(
    data_config: dict,
    shuffle: bool,
    columns: List[str],
    changing_columns: List[str],
    changed_columns: List[str],
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
        changing_name_columns=changing_columns,
        changed_name_columns=changed_columns,
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
        if changing_name_columns is not None:
            if not isinstance(changing_name_columns, list) or not isinstance(changed_name_columns, list):
                raise ValueError("changing_name_columns, changed_name_columns은 반드시 List 형태로 들어와야합니다.")
            
            if len(changing_name_columns) != len(changed_name_columns):
                raise ValueError("changing_name_columns와 changed_name_columns은 s`equential 해야하며 1대1 매칭되어야 합니다.")
            
            if any(item not in dataset.column_names for item in changing_name_columns):
                pass
            else:
                new_column_names = {old: new for old, new in zip(changing_name_columns, changed_name_columns)}
                dataset = dataset.rename_columns(new_column_names)
            
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns])
        
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

# data = get_datasets(data_config = DATA_CONFIG, shuffle = True, columns = ['question', 'chosen', 'rejected'], is_test = False)