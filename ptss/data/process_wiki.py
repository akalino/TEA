import json
import pandas as pd


if __name__ == "__main__":
    with open('enwiki-20160501/semantic-graphs-filtered-validation.02_06.json', 'rb') as f:
        val = json.load(f)
    print('Found {} validation instances'.format(len(val)))

    with open('enwiki-20160501/semantic-graphs-filtered-training.02_06.json', 'rb') as f:
        train = json.load(f)
    print('Found {} training instances'.format(len(train)))

    with open('enwiki-20160501/semantic-graphs-filtered-held-out.02_06.json', 'rb') as f:
        dev = json.load(f)
    print('Found {} hold-out instances'.format(len(dev)))

    all_triples = dev + val + train
    print('Total triples: {}'.format(len(all_triples)))
    print(all_triples[0])

    predicates = []
    sentences = []
    heads = []
    tails = []
    labels = []
    for j in val:
        p = j['edgeSet'][0]['kbID']
        predicates.append(p)
        s = ' '.join(j['tokens'])
        sentences.append(s)
        h = j['vertexSet'][0]['kbID']
        t = j['vertexSet'][1]['kbID']
        heads.append(h)
        tails.append(t)
        labels.append('val')
    print('Indexed validation')

    for j in train:
        p = j['edgeSet'][0]['kbID']
        predicates.append(p)
        s = ' '.join(j['tokens'])
        sentences.append(s)
        h = j['vertexSet'][0]['kbID']
        t = j['vertexSet'][1]['kbID']
        heads.append(h)
        tails.append(t)
        labels.append('train')
    print('Indexed training')

    for j in dev:
        p = j['edgeSet'][0]['kbID']
        predicates.append(p)
        s = ' '.join(j['tokens'])
        sentences.append(s)
        h = j['vertexSet'][0]['kbID']
        t = j['vertexSet'][1]['kbID']
        heads.append(h)
        tails.append(t)
        labels.append('holdout')
    print('Indexed holdout')

    unique_predicates = list(set(predicates))
    print('Found {} unique predicates'.format(len(unique_predicates)))

    unique_entities = list(set(heads + tails))
    print('Found {} unique entities'.format(len(unique_entities)))

    df = pd.DataFrame({'predicate_label': predicates,
                       'headID': heads,
                       'tailID': tails,
                       'sentence': sentences,
                       'split': labels})
    df.sort_values(by='predicate_label', inplace=True)

    train = df[df['split'] == 'train']
    print('Training: {}'.format(len(train)))
    triples_df = train[['headID', 'predicate_label', 'tailID']]
    triples_df.columns = ['s', 'p', 'o']
    triples_df.to_csv('wikidata_triples.csv', index=False)
    train_out = train[['headID', 'predicate_label', 'tailID']]
    train_out.to_csv('enwiki-20160501/train.txt', sep="\t",  header=None, index=False)
    train_sents = train[['sentence', 'predicate_label']]
    train_sents.to_csv('enwiki-20160501/train_sentences.txt', sep="\t", index=False)

    holdout = df[df['split'] == 'holdout']
    print('Holdout: {}'.format(len(holdout)))
    holdout_out = holdout[['headID', 'predicate_label', 'tailID']]
    holdout_out.to_csv('enwiki-20160501/test.txt', sep="\t",  header=None, index=False)
    hold_sents = holdout[['sentence', 'predicate_label']]
    hold_sents.to_csv('enwiki-20160501/hold_sentences.txt', sep="\t", index=False)

    validation = df[df['split'] == 'val']
    print('Validation: {}'.format(len(validation)))
    validation_out = validation[['headID', 'predicate_label', 'tailID']]
    validation_out.to_csv('enwiki-20160501/valid.txt', sep="\t", header=None, index=False)
    valid_sents = validation[['sentence', 'predicate_label']]
    valid_sents.to_csv('enwiki-20160501/valid_sentences.txt', sep="\t", index=False)
