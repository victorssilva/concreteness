import argparse
import logging
import os.path

import torch

import concreteness
import mirflickr

log = logging.getLogger(__name__)

DEFAULT_K = 50


def _setup_logging(verbose):
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging_format = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging_level, format=logging_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str, required=True,
                        help="Path to the directory of the mirflickr dataset.")
    parser.add_argument("-c", "--cache-dir", type=str, required=False,
                        help="Path to a directory to use for cache.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.",
                        action="store_true")
    parser.add_argument("-k", help="Number of neighbos to search for.", type=int,
                        default=DEFAULT_K, required=False)
    args = parser.parse_args()

    _setup_logging(args.verbose)

    images_directory = os.path.join(args.dataset_dir, "images")
    tags_directory = os.path.join(args.dataset_dir, "tags")

    vectors_file = None
    annoy_index_file = None
    if args.cache_dir is not None:
        vectors_file = os.path.join(args.cache_dir, "vectors.pt")
        annoy_index_file = os.path.join(args.cache_dir, "index.ann")

    log.info("Loading dataset.")
    dataset = mirflickr.MirflickrImagesDataset(images_directory, tags_directory,
                                               transform=concreteness.get_tensor_for_image)
    log.info("Dataset is loaded.")

    if vectors_file is not None and os.path.isfile(vectors_file):
        img_vectors = torch.load(vectors_file)
    else:
        log.info("Building image vectors.")
        img_vectors = concreteness.build_image_vectors(dataset)
        log.info("Built image vectors.")

        if vectors_file is not None:
            torch.save(img_vectors, vectors_file)
            log.info("Saved image vectors to %s", vectors_file)

    log.info("Computing concreteness.")
    nns = concreteness.build_nns(img_vectors, args.k, annoy_index_file=annoy_index_file)
    concreteness_dict = concreteness.get_concreteness(dataset, nns, args.k)
    log.info("Done!")

    sorted_concreteness = sorted(concreteness_dict.items(), key=lambda x: x[1], reverse=True)
    len(sorted_concreteness)

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
