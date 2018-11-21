import argparse
import logging
import math
import os.path

import mirflickr
import numpy as np
import torch
import torchvision.models as models
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from annoy import AnnoyIndex
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

cuda = torch.cuda.is_available()

log = logging.getLogger(__name__)


def show_image(dataset, image_index):
  image = dataset.get_pil_image(image_index)
  imshow(np.asarray(image))


# These are the expected values of the following pre-trained model.
# See https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models
input_image_size = (224, 224)
expected_mean = [0.485, 0.456, 0.406]
expected_std = [0.229, 0.224, 0.225]


def get_resnet_model():
  resnet = models.resnet50(pretrained=True)

  for param in resnet.parameters():
    param.requires_grad = False

  resnet.eval()

  if cuda:
    device = torch.device("cuda")
    resnet.to(device)

  return resnet


scaler = transforms.Resize(input_image_size)
normalize = transforms.Normalize(mean=expected_mean, std=expected_std)
to_tensor = transforms.ToTensor()


def img2vec(resnet, image_tensors):
  last_layer = resnet._modules.get("avgpool")

  if cuda:
      image_tensors = image_tensors.cuda()

  embedding = torch.zeros(image_tensors.shape[0], 2048, 1, 1)

  def copy_output(m, i, o):
      embedding.copy_(o.data)

  h = last_layer.register_forward_hook(copy_output)
  resnet(image_tensors)
  h.remove()

  return embedding


def get_tensor_for_image(image):
  variable = Variable(normalize(to_tensor(scaler(image))))
  return variable


def build_image_vectors(dataset, batch_size=64, num_workers=4):
  resnet = get_resnet_model()
  dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

  img_vectors = torch.zeros(len(dataset), 2048, 1, 1)
  number_of_batches = math.ceil(len(dataset) / batch_size)

  for batch_index, batch in enumerate(dataloader):
    batch_start = batch_index * batch_size
    batch_end = batch_start + batch_size

    # Pad the last batch if it's smaller than batch_size
    if batch.shape[0] != batch_size:
      new_batch = torch.zeros(batch_size, batch.shape[1], batch.shape[2],
                              batch.shape[3])
      new_batch[:batch.shape[0]] = batch
      img_vectors[batch_start:batch_end] = img2vec(resnet, new_batch)[:batch.shape[0]]
      continue

    img_vectors[batch_start:batch_end] = img2vec(resnet, batch)
    log.debug("Computed batch %s out of %s", batch_index, number_of_batches)

  return img_vectors


def get_images_by_tag(filtered_tags):
  images_by_tag = {}
  for image, tags in filtered_tags.items():
    for tag in tags:
      images_by_tag.setdefault(tag, set()).add(image)

  return images_by_tag


def build_annoy_index(img_vectors):
  annoy_index = AnnoyIndex(2048)
  for i in range(len(img_vectors)):
    annoy_index.add_item(i, img_vectors[i])

  annoy_index.build(10)
  return annoy_index


def load_annoy_index(file):
  annoy_index = AnnoyIndex(2048)
  annoy_index.load(file)
  return annoy_index


def get_concreteness_for_word(word, associated_images, filtered_image_to_index, annoy_index_file,
                              n, k):
  annoy_index = load_annoy_index(annoy_index_file)
  mni = 0.0

  for image in associated_images:
    if image not in filtered_image_to_index:
      continue

    neighbors = set(annoy_index.get_nns_by_item(filtered_image_to_index[image], k))
    mni += 1.0 * (len(associated_images.intersection(neighbors)))

  mni = mni / len(associated_images)
  denominator = (1.0 * k * len(associated_images)) / n
  return mni / denominator


def get_concreteness(dataset, annoy_index_file, k=50):
  images_by_tag = get_images_by_tag(dataset.filtered_tags)
  filtered_image_to_index = dataset.get_filtered_image_to_index()
  n = len(dataset.filtered_tags)

  with mp.Pool() as pool:
    results = {}
    for word in images_by_tag:
      associated_images = images_by_tag[word]
      results[word] = pool.apply_async(get_concreteness_for_word,
                                       (word, associated_images, filtered_image_to_index,
                                        annoy_index_file, n, k))

  concreteness = {word: result.get() for (word, result) in results.items()}
  return concreteness


def get_concreteness_for_word_precomputed_nns(word, associated_images, filtered_image_to_index,
                                              nns, n, k):
  mni = 0.0

  for image in associated_images:
    if image not in filtered_image_to_index:
      continue

    neighbors = nns[filtered_image_to_index[image]]
    mni += 1.0 * (len(associated_images.intersection(neighbors)))

  mni = mni / len(associated_images)
  denominator = (1.0 * k * len(associated_images)) / n
  return mni / denominator


def get_concreteness_precomputed_nns(dataset, nns, k):
  images_by_tag = get_images_by_tag(dataset.filtered_tags)
  filtered_image_to_index = dataset.get_filtered_image_to_index()
  n = len(dataset.filtered_tags)

  concreteness = {}
  i = 0
  for word in images_by_tag:
    associated_images = images_by_tag[word]
    concreteness[word] = get_concreteness_for_word_precomputed_nns(
      word, associated_images, filtered_image_to_index, nns, n, k)
    if not i % 1000:
      log.debug("Done with word %s out of %s", i, len(images_by_tag))
    i += 1
  return concreteness


def build_nns(img_vectors, k, annoy_index_file=None):
  if annoy_index_file is not None and os.path.isfile(annoy_index_file):
    annoy_index = load_annoy_index(annoy_index_file)
    log.info("Loaded annoy index.")
  else:
    log.info("Building annoy index.")
    annoy_index = build_annoy_index(img_vectors)

    if annoy_index_file is not None:
      annoy_index.save(annoy_index_file)
      log.info("Annoy index was saved to %s.", annoy_index_file)

  log.info("Bulding NNS map.")
  nns = {}
  for index in range(len(img_vectors)):
    nns[index] = set(annoy_index.get_nns_by_item(index, k))
    if not index % 1000:
      log.debug("Done saving NNS for %s out of %s", index, len(img_vectors))

  return nns


def _setup_logging(verbose):
  logging_level = logging.DEBUG if verbose else logging.INFO
  logging_format='%(asctime)s [%(levelname)s] %(message)s'
  logging.basicConfig(level=logging_level, format=logging_format)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--dataset-dir", type=str, required=True,
                      help="Path to the directory of the mirflickr dataset.")
  parser.add_argument("-c", "--cache-dir", type=str, required=False,
                      help="Path to a directory to use for cache.")
  parser.add_argument("-v", "--verbose", help="Increase output verbosity.",
                      action="store_true")
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
                                             transform=get_tensor_for_image)
  log.info("Dataset is loaded.")

  if vectors_file is not None and os.path.isfile(vectors_file):
    img_vectors = torch.load(vectors_file)
  else:
    log.info("Building image vectors.")
    img_vectors = build_image_vectors(dataset)
    log.info("Built image vectors.")

    if vectors_file is not None:
      torch.save(img_vectors, vectors_file)
      log.info("Saved image vectors to %s", vectors_file)

  log.info("Computing concreteness.")
  k = 50
  nns = build_nns(img_vectors, k, annoy_index_file=annoy_index_file)
  concreteness = get_concreteness_precomputed_nns(dataset, nns, k)
  log.info("Done!")

  sorted_concreteness = sorted(concreteness.items(), key=lambda x: x[1], reverse=True)
  len(sorted_concreteness)

  from IPython import embed
  embed()


if __name__ == "__main__":
  main()
