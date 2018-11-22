import json
import logging
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def get_tags(tags_file):
    with open(tags_file, encoding="utf8") as f:
        return list(map(lambda x: x.strip(), f.readlines()))


def get_index_from_tags_file(tags_file):
    return int(tags_file.split("/")[-1][:-4])


def build_tags_json(tag_files, tags_json_filename):
    image_tags = {}

    for i, tags_file in enumerate(tag_files):
        image_tags[get_index_from_tags_file(tags_file)] = get_tags(tags_file)

    with open(tags_json_filename, "w", encoding="utf8") as tags_json_file:
        json.dump(image_tags, tags_json_file)

    return image_tags


def load_tags_json(tags_json_filename):
    with open(tags_json_filename, "r") as tags_json_file:
        image_tags = json.load(tags_json_file)
        return image_tags


def get_tag_scores(image_tags):
    tag_scores = {}
    for tags in image_tags.values():
        for tag in tags:
            tag = tag.lower()
            tag_scores[tag] = tag_scores.get(tag, 0) + 1

    return tag_scores


def get_filtered_tags(image_tags):
    tag_scores = get_tag_scores(image_tags)

    # Will keep only the tags that have at least 200 occurrences
    filtered_tags = {}
    for image, tags in image_tags.items():
        filtered_image_tags = []
        for tag in tags:
            if tag_scores.get(tag, 0) >= 200:
                filtered_image_tags.append(tag)

        if filtered_image_tags:
            filtered_tags[int(image)] = filtered_image_tags

    return filtered_tags


def list_tag_files(tags_directory):
    tag_files = []
    for subdir in os.listdir(tags_directory):
        subdir_files = os.listdir(os.path.join(tags_directory, subdir))
        for file in subdir_files:
            tag_files.append(os.path.join(tags_directory, subdir, file))

    return sorted(tag_files, key=lambda f: get_index_from_tags_file(f))


def get_image_score(image_tags, image_index):
    return len(image_tags.get(image_index, []))


def get_filtered_images(filtered_tags, image_indexes, min_score=3):
    return [img for img in image_indexes if get_image_score(filtered_tags, img) >= min_score]


class MirflickrImagesDataset(Dataset):
    dataset_size = 1000000
    subdirectory_size = 10000

    def __init__(self, images_directory, tags_directory, transform=None):
        self.images_directory = images_directory
        self.transform = transform

        image_indexes = list(range(self.dataset_size))

        tags_json_filename = os.path.join(tags_directory, "tags.json")
        try:
            image_tags = load_tags_json(tags_json_filename)
        except FileNotFoundError:
            image_tags = build_tags_json(list_tag_files(tags_directory), tags_json_filename)

        self.filtered_tags = get_filtered_tags(image_tags)
        self.filtered_images = get_filtered_images(self.filtered_tags, image_indexes)

    def __len__(self):
        return len(self.filtered_images)

    def get_image_path(self, image_index):
        image_subdirectory = str(int(image_index / self.subdirectory_size))
        return os.path.join(self.images_directory, image_subdirectory, "{}.jpg".format(image_index))

    def get_pil_image(self, index):
        return Image.open(self.get_image_path(self.filtered_images[index]))

    def __getitem__(self, index):
        image = self.get_pil_image(index)
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                log.warning("Could not transform image %s due to %s. Skipping.", index, e)
                return torch.zeros(3, 224, 224)

        return image

    def get_filtered_image_to_index(self):
        filtered_image_to_index = {}
        for i, image_index in enumerate(self.filtered_images):
            filtered_image_to_index[image_index] = i
        return filtered_image_to_index
