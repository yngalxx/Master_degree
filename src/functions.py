import csv


def from_tsv_to_list(path, delimiter="\n"):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)

    expected = list(read_tsv)

    return [item for sublist in expected for item in sublist]


def collate_fn(batch):
    return tuple(zip(*batch))
