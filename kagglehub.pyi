from typing import overload

@overload
def dataset_download(dataset: str) -> str: ...

@overload
def dataset_download(dataset: str, path: str) -> str: ...
