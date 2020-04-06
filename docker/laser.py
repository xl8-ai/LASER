#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
from pathlib import Path
from LASER.source.lib.text_processing import Token, BPEfastApply
from LASER.source.embed import *
import functools
import numpy as np
import torch.nn as nn
import torch
import argparse


def _strip_unnecessary_characters(source: object) -> object:
    """Remove unnecessary characters from an input string."""
    return source.replace("\r ", " ").replace("\r", " ").rstrip("\n")


def check_embed_and_load_if_need(func):
    """Decorate USE methods to make sure we load a USE model before using it."""
    def wrapper(self, *args, **kwargs):
        if self.encoder is None:
            self.load_encoder()
        return func(self, *args, **kwargs)

    return functools.update_wrapper(wrapper, func)


def compute_cosine_distances(a, b):
    # x shape is n_a * dim
    # y shape is n_b * dim
    # results shape is n_a * n_b
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return round(float(cos(torch.from_numpy(a), torch.from_numpy(b))[0]), 4)


class AlignmentUtils:
    """Utility Class to use USE model."""

    encoder = None
    model_dir = Path(__file__).parent / "models"
    encoder_path = model_dir / "bilstm.93langs.2018-12-26.pt"
    bpe_codes_path = model_dir / "93langs.fcodes"

    def load_encoder(self):
        print(f' - Encoder: loading {self.encoder_path}')
        self.encoder = SentenceEncoder(self.encoder_path,
                                       max_sentences=None,
                                       max_tokens=12000,
                                       sort_kind='mergesort')

    def embed_individual_line(self, content, lang):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ifname = tmpdir / "content.txt"
            bpe_fname = tmpdir / 'bpe'
            bpe_oname = tmpdir / 'out.raw'
            with ifname.open("w") as f:
                f.write(content)
            if lang != '--':
                tok_fname = tmpdir / "tok"
                Token(str(ifname),
                      str(tok_fname),
                      lang=lang,
                      romanize=True if lang == 'el' else False,
                      lower_case=True,
                      gzip=False,
                      verbose=True,
                      over_write=False)
                ifname = tok_fname
            BPEfastApply(str(ifname),
                         str(bpe_fname),
                         str(self.bpe_codes_path),
                         verbose=True, over_write=False)
            ifname = bpe_fname
            EncodeFile(self.encoder,
                       str(ifname),
                       str(bpe_oname),
                       verbose=True,
                       over_write=False,
                       buffer_size=10000)
            dim = 1024
            X = np.fromfile(str(bpe_oname), dtype=np.float32, count=-1)
            X.resize(X.shape[0] // dim, dim)
        return X

    @check_embed_and_load_if_need
    def distinguish_pair_sentence(self, source, target, lang1, lang2, threshold=0.5):
        """
        Distinguish pair sentence using the MUSE model.

        :param source: UTF-8 encoded source sentence
        :param target: UTF-8 encoded target sentence
        :param threshold: threshold to determine whether those two are good pair or not.
        :return: True or False with the result ratio - True if the result is better than a given ratio.
        """
        embedded_dict = {}

        input_dict = {
            lang1: source,
            lang2: target
        }
        for key in input_dict:
            embedded_dict[key] = self.embed_individual_line(input_dict[key], lang=key)
        ratio = compute_cosine_distances(embedded_dict[lang1], embedded_dict[lang2])
        if ratio >= threshold:
            return True, ratio
        return False, ratio

    def distinguish_pair_sentences(self, source_it, target_it, threshold, langs):
        """
        Distinguish pair sentences with modelalignment_utils.py.

        :param source_it: lang1 file iterator
        :param target_it: lang2 file iterator
        :param threshold: threshold for USE result comparison
        """
        lang1 = langs[0]
        lang2 = langs[1]
        for source_line, target_line in zip(source_it, target_it):
            try:
                source_sentence = _strip_unnecessary_characters(source_line.decode("utf-8"))
                target_sentence = _strip_unnecessary_characters(target_line.decode("utf-8"))
                _, ratio = self.distinguish_pair_sentence(source_sentence, target_sentence, lang1, lang2, threshold=threshold)
                yield source_sentence, target_sentence, ratio
            except UnicodeDecodeError:
                print(f"UTF-8 Decode error from {source_sentence}, {target_sentence} - skip this.")


def start_use_batching(alignment_utils, args):
    """
    Download, Open, Read, Save data.

    :param alignment_utils:
    :return:
    """
    try:
        source_it = open(args.input_lang1, "rb")
        target_it = open(args.input_lang2, "rb")
        source_output_handle = open(args.output_lang1, "w", encoding="utf-8")
        target_output_handle = open(args.output_lang2, "w", encoding="utf-8")
        good_output_handle = open(args.output_good, "w", encoding="utf-8")
        bad_output_handle = open(args.output_bad, "w", encoding="utf-8")

        for source_sentence, target_sentence, ratio in \
                alignment_utils.distinguish_pair_sentences(source_it, target_it, args.ratio, args.langs):
            result_txt = f"{ratio}|{source_sentence}|{target_sentence}\n"
            if ratio >= args.ratio:
                good_output_handle.write(result_txt)
                source_output_handle.write(f"{source_sentence}\n")
                target_output_handle.write(f"{target_sentence}\n")
            else:
                bad_output_handle.write(result_txt)

    except FileNotFoundError:
        print("FileNotFoundError exception - Please check input files.")
    else:
        source_it.close()
        target_it.close()
        source_output_handle.close()
        target_output_handle.close()
        good_output_handle.close()
        bad_output_handle.close()


def parse_args(args):
    """
    Pass parameters.

    :return: parser
    """
    parser = argparse.ArgumentParser(description='Align sentences with "use"')
    parser.add_argument(
        "--input_lang1", type=str, help="Path for lang1 input file", required=True)
    parser.add_argument(
        "--input_lang2", type=str, help="Path for lang2 input file", required=True)
    parser.add_argument(
        "--output_lang1", type=str, help="Path for lang1 output file", required=True)
    parser.add_argument(
        "--output_lang2", type=str, help="Path for lang2 output file", required=True)
    parser.add_argument(
        "--output_good", type=str, help="Path for good sentence pair  file", required=True)
    parser.add_argument(
        "--output_bad", type=str, help="Path for bad sentence pair file", required=True)
    parser.add_argument(
        "--ratio", type=float, help="The ratio for the threshold", required=False, default=0.2)
    parser.add_argument('langs', nargs=2)

    return parser.parse_args(args)