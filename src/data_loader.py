import pandas as pd
import os
from pathlib import Path
from typing import Optional,Union
from error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS

data_reader_functions = {".csv": pd.read_csv, ".parquet": pd.read_parquet}



class DataLoader:
    
    def load_data(self,file_path:Union[str, Path]) -> Optional[pd.DataFrame]:
        
        self._validate_file_path(file_path)
        ext = self._check_if_file_extension_supported(file_path)

        reader_func = data_reader_functions.get(ext)
        data: pd.DataFrame = reader_func(file_path)

        if data.empty:
            raise ValueError(EM.EMPTY_DATA_FILE.value)
        
        return data

    
    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
                if not isinstance(file_path, (str,Path)):
                    
                    raise TypeError(
                EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path))
            )
        
                if not os.path.exists(file_path):
                    raise FileNotFoundError(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
            
    
    def _check_if_file_extension_supported(self, file_path: Union[str, Path]) -> str:

                ext = Path(file_path).suffix

                if ext not in SUPPORTED_FILE_EXTENSIONS:
                    raise ValueError(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )   
                return ext
            

