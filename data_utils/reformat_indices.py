

def read_predicate_map(_path):
    """
    Loads predicate to index map.

    :param _path: Path to text file.
    :return: Dictionary with indices as keys, string names as values.
    """
    _map = {}
    with open(_path) as f:
        for line in f:
            vs = line.split("\t")
            if len(vs) > 1:
                _map[vs[1].strip("\n")] = vs[0]
    return _map


def read_entity_map(_path):
    """
    Loads predicate to index map.

    :param _path: Path to text file.
    :return: Dictionary with indices as keys, string names as values.
    """
    _map = {}
    with open(_path) as f:
        for line in f:
            vs = line.split("\t")
            if len(vs) > 1:
                _map[vs[1].strip("\n")] = vs[0]
    return _map


if __name__ == "__main__":
    rel_out = {}
    with open('relation_ids.del', 'r') as f:
        for l in f.readlines():
            rel_out[l.split('\t')[1].strip('\n')] = int(l.split('\t')[0])
    with open('relation2id.txt', 'w') as g:
        for j in rel_out.keys():
            g.write('{}\t{}\n'.format(j, rel_out[j]))
    ent_out = {}
    with open('entity_ids.del', 'r') as f:
        for l in f.readlines():
            ent_out[l.split('\t')[1].strip('\n')] = int(l.split('\t')[0])
    with open('entity2id.txt', 'w') as g:
        for j in ent_out.keys():
            g.write('{}\t{}\n'.format(j, ent_out[j]))
