# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

_ENIT_TRAIN_DATASETS = [
    [
        "http://opus.nlpl.eu/download.php?f=News-Commentary11%2Fen-it.txt.zip",  # pylint: disable=line-too-long
        ("en-it.txt/News-Commentary.en-it.en",
         "en-it.txt/News-Commentary.en-it.it")
    ]
]
_ENIT_TEST_DATASETS = [
    [
        "http://opus.nlpl.eu/download.php?f=OpenSubtitles%2Fen-it.txt.zip",
     ("en-it.txt\ \(1\)/OpenSubtitles.en-it.en", "en-it.txt\ \(1\)/OpenSubtitles.en-it.it")
    ]
]


def _get_wmt_enit_bpe_dataset(directory, filename):
  """Extract the WMT en-it corpus `filename` to directory unless it's there."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + ".it") and
          tf.gfile.Exists(train_path + ".en")):
    url = ("https://drive.google.com/open?id=1F3apMpe1lijbUzZfMNPBJlvURgV3Sx2t")
    corpus_file = generator_utils.maybe_download_from_drive(
        directory, "News-Commentary11-enit.zip", url)
    with zipfile.open(corpus_file, "r") as corpus_zip:
      corpus_zip.extractall(directory)
  return train_path


@registry.register_problem
class TranslateEnitWmtBpe32k(translate.TranslateProblem):
  """Problem spec for WMT En-It translation, BPE version."""

  @property
  def approx_vocab_size(self):
    return 32000

  @property
  def vocab_filename(self):
    return "vocab.bpe.%d" % self.approx_vocab_size

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_filename) and force_get:
      raise ValueError("Vocab %s not found" % vocab_filename)
    return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the WMT en->de task, training set."""
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset_path = ("train.tok.clean.bpe.32000"
                    if train else "newstest2013.tok.bpe.32000")     #da controllare
    train_path = _get_wmt_enit_bpe_dataset(tmp_dir, dataset_path)

    # Vocab
    token_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(token_path):
      token_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
      tf.gfile.Copy(token_tmp_path, token_path)
      with tf.gfile.GFile(token_path, mode="r") as f:
        vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
      with tf.gfile.GFile(token_path, mode="w") as f:
        f.write(vocab_data)

    return text_problems.text2text_txt_iterator(train_path + ".en",
                                                train_path + ".it")


@registry.register_problem
class TranslateEnitWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def vocab_filename(self):
    return "vocab.enit.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENIT_TRAIN_DATASETS if train else _ENIT_TEST_DATASETS


@registry.register_problem
class TranslateEnitWmt8kPacked(TranslateEnitWmt8k):

  @property
  def packed_length(self):
    return 256


@registry.register_problem
class TranslateEnitWmtCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
