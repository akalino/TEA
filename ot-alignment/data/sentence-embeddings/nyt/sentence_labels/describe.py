import pandas as pd


def read_files():
    train = pd.read_csv('training_data.csv')
    valid = pd.read_csv('validation_data.csv')
    test = pd.read_csv('testing_data.csv')
    print(len(train))
    print(len(valid))
    print(len(test))
    train_sent = list(set(train['sentence'].tolist()))
    valid_sent = list(set(valid['sentence'].tolist()))
    test_sent = list(set(test['sentence'].tolist()))
    print('{}-{}-{}'.format(len(train_sent),
                            len(valid_sent),
                            len(test_sent)))
    all_sent = list(set(train_sent + valid_sent + test_sent))
    print(len(all_sent))
    train['triple_string'] = train.apply(lambda x: str(x['subject_id']) +
                                                   str(x['object_id']) +
                                                   str(x['rel_idx']), axis=1)
    valid['triple_string'] = valid.apply(lambda x: str(x['subject_id']) +
                                                   str(x['object_id']) +
                                                   str(x['rel_idx']), axis=1)
    test['triple_string'] = test.apply(lambda x: str(x['subject_id']) +
                                                 str(x['object_id']) +
                                                 str(x['rel_idx']), axis=1)
    train_triples = list(set(train['triple_string'].tolist()))
    valid_triples = list(set(valid['triple_string'].tolist()))
    test_triples = list(set(test['triple_string'].tolist()))
    print('{}-{}-{}'.format(len(train_triples),
                            len(valid_triples),
                            len(test_triples)))
    all_triples = list(set(train_triples + valid_triples + test_triples))
    print(len(all_triples))
    #train = train[train['rel_idx'] != 35]
    train.sort_values(by='sent_id', inplace=True, ignore_index=True)
    train2 = train[['sent_id', 'triple_string', 'sentence']]
    print(train2.head(30))

    train.sort_values(by='triple_string', inplace=True, ignore_index=True)
    train2 = train[['sent_id', 'triple_string', 'sentence']]
    print(train2.head(82))


if __name__ == "__main__":
    read_files()
