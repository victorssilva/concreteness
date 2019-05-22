# Concreteness
An implementation of [Quantifying the Visual Concreteness of Words and Topics in Multimodal Datasets](http://www.cs.cornell.edu/~jhessel/concreteness/paper.pdf) with PyTorch.

It uses a ResNet50 along with Spotify's [Annoy](https://github.com/spotify/annoy) library to compute the visual concreteness scores of words from [MIRFLICKR](http://press.liacs.nl/mirflickr/).

## Requirements
To install the basic requirements, run this:

`pip install -r requirements.txt`

If you'd like use a Jupyter Notebook for interacting with the concreteness scores after computing them, you'll also need:

`pip install -r requirements-notebook.txt`

As of now, the existing code has only been tested with Python3.6 and Python 3.7.

For the MSCOCO dataset, you'd also have to download the English model for SpaCy by 

`python -m spacy download en`

## Usage

### Downloading the dataset
Before running, you'll need to download the MIRFLICKR dataset. You can do that with:

```
cd data
./get_mirflickr.sh
```

It's 120GB, so it may take a while.

Similarly, you can get the MSCOCO dataset with:

```
cd data
./get_mscoco.sh
```

### Shell usage
Once your download is finished, you can compute the concreteness scores with:

`python main.py -d <mirflickr_directory> -c <cache_directory> -v`

Swap in the path to where the mirflickr dataset was downloaded to and a directory of your choice to use for caching.

For the MSCOCO dataset, run with 

`python main.py -d <mscoco_directory> -c <cache_directory> -v -t mscoco`

### Jupyter Notebook

If you prefer, you can also run the provided Jupyter Notebook:

`jupyter notebook concreteness.ipynb`

## TODO
- Improve Jupyter Notebook formatting

## Thanks to
[@jmhessel](https://github.com/jmhessel/) for helpful pointers and a great paper.

Citation:

```
@inproceedings{hessel2018concreteness,
               title={Quantifying the visual concreteness of words and topics in multimodal datasets},
               author={Hessel, Jack and Mimno, David and Lee, Lillian},
               booktitle={NAACL},
               year={2018}
}
```
