import csv
import json
from typing import Dict, List


def dump_json(path: str, dict_to_save: Dict) -> None:
    jsonString = json.dumps(dict_to_save, indent=4)
    with open(path, "w") as jsonFile:
        jsonFile.write(jsonString)


def from_tsv_to_list(
    path: str, delimiter: str = "\n", skip_empty_lines: bool = True
) -> List:
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)
    expected = list(read_tsv)
    if skip_empty_lines:
        return [item for sublist in expected for item in sublist]
    with_empty_lines = []
    for sublist in expected:
        if len(sublist) == 0:
            with_empty_lines.append("")
        else:
            with_empty_lines.extend(iter(sublist))
    return with_empty_lines


def save_list_to_tsv_file(path: str, output_list: List) -> None:
    with open(path, "w", newline="") as f_output:
        tsv_output = csv.writer(f_output, delimiter="\n")
        tsv_output.writerow(output_list)
