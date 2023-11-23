import argparse

from trainer import experiment


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--source-path',
        help='GCS file or local paths to source embeddings',
        nargs='+',
        default='gs://triple_vectors')
    PARSER.add_argument(
        '--target-path',
        help='GCS file or local path to target embeddings',
        nargs='+',
        default='gs://sent-embeddings')
    PARSER.add_argument(
        '--dataset',
        help='Dataset name for distant supervision',
        nargs='+',
        default='nytfb'
    )
    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default='/tmp/bigot_outputs')
    args = PARSER.parse_args()
    experiment.run(args)
