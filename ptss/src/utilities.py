import os
from google.cloud import storage
from google.api_core.exceptions import Conflict
from tqdm import tqdm

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'triple-storage-key.json'


def download_blob_file(bucket_name, blob_name, dest):
    """Downloads a blob into memory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest)


def archive_model(_model_path):
    """

    :param _model_path: triple_vectors/triples_fb15k-237-complex-avg_300.pt
    :return:
    """
    client = storage.Client()
    try:
        bucket = client.create_bucket('triple_vectors')
    except Conflict:
        bucket = client.get_bucket('triple_vectors')
    new_blob = bucket.blob('{}'.format(_model_path.split('/')[1]))
    new_blob.upload_from_filename(filename=_model_path)


def fetch_model(_model_path):
    """

    :param _model_path: triple_vectors/triples_fb15k-237-complex-avg_300.pt
    :return:
    """
    wd = os.path.normpath(os.getcwd())
    dest_fname = os.path.join(wd, _model_path)
    print('Saving model to {}'.format(dest_fname))
    download_blob_file('triple_vectors', '{}'.format(_model_path.split('/')[1]), dest_fname)


def fetch_ptss_assets(_model_name):
    wd = os.path.normpath(os.getcwd())
    for _a in tqdm(['dev', 'test', 'training', 'weights']):
        dest_fname = os.path.join(os.path.join(wd, 'test'), '{}-{}.npy'.format(_model_name, _a))
        print(dest_fname)
        download_blob_file('intermediate_assets', '{}-{}'.format(_model_name, _a), dest_fname)


def archive_ptss_assets(_model_name):
    """

    :param _model_name: has format {dataset}-{kg-type}-{agg-type}
    :return:
    """
    wd = os.path.normpath(os.getcwd())
    client = storage.Client()
    try:
        bucket = client.create_bucket('intermediate_assets')
    except Conflict:
        bucket = client.get_bucket('intermediate_assets')
    for _a in tqdm(['dev', 'test', 'training', 'weights']):
        new_blob = bucket.blob('{}-{}'.format(_model_name, _a))
        new_blob.upload_from_filename(filename=os.path.join(os.path.join(wd, 'intermediate'),
                                                            '{}-{}.npy'.format(_model_name, _a)))


def bulk_upload():
    models = ['rotate', 'transe']
    aggs = ['avg', 'had', 'ht', 'l1', 'l2']
    ns = [10, 20, 50]
    for n in ns:
        for m in models:
            for a in aggs:
                try:
                    archive_model('triple_vectors/triples_fb15k-237-{}-{}_300_{}.pt'.format(m, a, n))
                    print('Completed model upload {}-{}-{}...'.format(m, a, n))
                except FileNotFoundError:
                    print('model {}-{}-{} not trained yet...'.format(m, a, n))
                #archive_ptss_assets('fb15k-237-{}-{}-{}'.format(m, a, n))
                #print('Completed asset uploads {}-{}-e}...'.format(m, a, n))


if __name__ == "__main__":
    bulk_upload()
