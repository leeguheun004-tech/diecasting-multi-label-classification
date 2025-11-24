import os
from dataclasses import dataclass, field
from datetime import datetime
import os.path as osp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
now = datetime.now()

@dataclass
class BaseConfig:
    mode: bool  = 'test'


@dataclass
class DataConfig:
    label_csv_name: str     = 'diecasting_w_imc'
    label_list: list[str]   = field(
        default_factory=lambda: [
            'P', 
            'S',
            "IMC"
        ])
    bit_labels: dict[str, str] = field(
        default_factory=lambda: {
        "0": "Normal",
        "4": "P",
        "2": "S",
        "6": "PS",
        "1": "IMC",
        "3": "S_IMC",
        "5": "P_IMC",
        "7": "PS_IMC",
        })
    
    data_dir: Path   = Path(osp.join(os.getcwd(), "dataset"))
    
@dataclass
class TrainConfig:
    arch_name: str   = "resnet101"
    model_name: str  = f'v2_{arch_name}_v1.1.1'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{model_name}"
    code_dir: Path   = BASE_DIR / "experiments" / "codes" / f"{model_name}"
    
    num_epochs: int  = 100
    batch_size: int  = 16
    workers: int     = 8
    lr: float        = 1e-4
    
    num_classes: int = 3
    train_thld: float = 0.5
    

@dataclass
class TestConfig:
    arch_name: str   = "resnet101"
    model_name: str  = f'v2_{arch_name}_v1.1.1'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{model_name}"
    
    batch_size: int = 16
    epoch: int = 99
    threshold: float = 0.5
    workers: int = 8
    
    mispred_detail: str = 'IMC'
