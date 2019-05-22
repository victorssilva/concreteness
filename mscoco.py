import json
import logging
import os
import spacy
import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
nlp = spacy.load('en')

def get_tags(sentence):
    return [w.text for w in nlp(sentence.strip())]


def get_index_from_annotation(image_id_2_index, annotation):
    return image_id_2_index[annotation['image_id']]


def build_tags_json(annotation_path, tags_json_filename):
    image_tags = dict()
    image_paths = list()
    annotations = json.load(open(annotation_path))
    image_id_2_index = dict()

    for i, image in enumerate(annotations['images']):
        image_id_2_index[image['id']] = i
        image_paths.append(image['file_name'])
    for i, annotation in enumerate(annotations['annotations']):
        if i % 1000 == 999:
            print('Processed {:d}/{:d} annotations.'.format(i+1, len(annotations['annotations'])))
        image_index_new = get_index_from_annotation(image_id_2_index, annotation)
        if image_index_new not in image_tags:
            image_tags[image_index_new] = list()
        image_tags[image_index_new].extend(get_tags(annotation['caption']))

    with open(tags_json_filename, 'w', encoding='utf8') as tags_json_file:
        json.dump((image_paths, image_tags), tags_json_file)

    return image_paths, image_tags


def load_tags_json(tags_json_filename):
    with open(tags_json_filename, 'r') as tags_json_file:
        image_paths, image_tags = json.load(tags_json_file)
        return image_paths, image_tags


def get_tag_scores(image_tags):
    tag_scores = dict()
    for tags in image_tags.values():
        for tag in tags:
            tag = tag.lower()
            tag_scores[tag] = tag_scores.get(tag, 0) + 1

    return tag_scores


def get_filtered_tags(image_tags):
    tag_scores = get_tag_scores(image_tags)

    # Will keep only the tags that have at least 200 occurrences
    filtered_tags = dict()
    for image_idx, tags in image_tags.items():
        filtered_image_tags = []
        for tag in tags:
            if tag_scores.get(tag, 0) >= 200:
                filtered_image_tags.append(tag)

        if filtered_image_tags:
            filtered_tags[image_idx] = filtered_image_tags

    return filtered_tags


def get_image_score(image_tags, image_index):
    return len(image_tags.get(image_index, []))


def get_filtered_images(filtered_tags, image_indexes, min_score=3):
    return [img for img in image_indexes if get_image_score(filtered_tags, img) >= min_score]


def get_images_by_tag(tags):
    images_by_tag = dict()
    for image_id, tags in tags.items():
        for tag in tags:
            images_by_tag.setdefault(tag, set()).add(image_id)

    return images_by_tag


class MSCOCODataset(Dataset):

    def __init__(self, images_directory, annotation_path, transform=None):
        self.images_directory = images_directory
        self.transform = transform
        
        tags_json_filename = os.path.join(images_directory, "tags.json")
        try:
            self.image_paths, image_tags = load_tags_json(tags_json_filename)
        except FileNotFoundError:
            self.image_paths, image_tags = build_tags_json(annotation_path, tags_json_filename)

        image_indexes = list(range(len(self.image_paths)))

        filtered_tags = get_filtered_tags(image_tags)
        self.filtered_images = get_filtered_images(filtered_tags, image_indexes)

        # Because we've filtered out some of our images, we'll now map the previous image indices
        # to the new ones to avoid dealing with gaps.
        self.filtered_tags = self._build_filtered_tags_with_converted_image_indexes(filtered_tags)
        self.images_by_tag = get_images_by_tag(self.filtered_tags)

    def __len__(self):
        return len(self.filtered_images)

    def get_image_path(self, image_index):
        return os.path.join(self.images_directory, self.image_paths[image_index])

    def get_pil_image(self, index):
        return Image.open(self.get_image_path(self.filtered_images[index])).convert('RGB')

    def __getitem__(self, index):
        image = self.get_pil_image(index)
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                log.warning("Could not transform image %s due to %s. Skipping.", index, e)
                return torch.zeros(3, 224, 224)

        return image

    def _build_filtered_image_to_index_map(self):
        filtered_image_to_index = dict()
        filtered_image_paths = list()
        for i, image_index in enumerate(self.filtered_images):
            filtered_image_to_index[image_index] = i
            filtered_image_paths.append(self.image_paths[image_index])
        self.image_paths = filtered_image_paths
        return filtered_image_to_index

    def _build_filtered_tags_with_converted_image_indexes(self, filtered_tags):
        filtered_image_to_index = self._build_filtered_image_to_index_map()
        filtered_tags_with_converted_image_indexes = dict()
        for image, tags in filtered_tags.items():
            if image not in filtered_image_to_index:
                continue
            new_image_index = filtered_image_to_index[image]
            filtered_tags_with_converted_image_indexes[new_image_index] = tags

        return filtered_tags_with_converted_image_indexes
