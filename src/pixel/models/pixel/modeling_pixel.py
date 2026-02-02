
"""
PyTorch PIXEL models
"""

import collections
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
np.random.seed(42) #TODO: Erase
import torch
import wandb
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ViTForImageClassification, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ...utils import DependencyParsingModelOutput, format_mask
from ..biaffine import Biaffine
from ..pooling import PoolingForSequenceClassificationHead, PoolingMode
from ..vit import ViTModel
from .configuration_pixel import PIXELConfig

logger = logging.getLogger(__name__)


PIXEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Team-PIXEL/pixel-base",
]


class PIXELForBiaffineParsing(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=True)

        self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        word_starts=None,
        head_mask=None,
        arc_labels=None,
        rel_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r""" """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # wandb.log({"pixel_values": wandb.Image(pixel_values)})

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        outs = self.dropout(outputs[0][:, 1:, :])

        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        word_outputs_heads = torch.cat([outputs[1].unsqueeze(1), word_outputs_deps], dim=1)

        arc_logits = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_logits = arc_logits.squeeze()

        rel_logits = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_logits = rel_logits.permute(0, 2, 3, 1)

        loss = None
        if arc_labels is not None and rel_labels is not None:
            loss = self._get_loss(arc_logits, rel_logits, arc_labels, rel_labels, self.loss_fn)

        if len(arc_logits.shape) == 2:
            arc_logits = arc_logits.unsqueeze(0)

        if not return_dict:
            output = (
                arc_logits,
                rel_logits,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DependencyParsingModelOutput(
            loss=loss,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.config.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                if start == end:
                    vecs_range = subword_vecs[start]
                    word_vecs.append(vecs_range.unsqueeze(0))
                else:
                    vecs_range = subword_vecs[start:end]
                    word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.config.hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to(self.device)

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.config.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss


class PIXELForTokenClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output[:, 1:, :])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PIXELForSequenceClassification(ViTForImageClassification):
    def __init__(self, config, pooling_mode: PoolingMode = PoolingMode.MEAN, add_layer_norm: bool = True):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels

        self.add_cls_pooling_layer = pooling_mode == PoolingMode.CLS
        self.vit = ViTModel(config, add_pooling_layer=self.add_cls_pooling_layer)

        # Classifier head
        self.pooler = PoolingForSequenceClassificationHead(
            hidden_size=config.hidden_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            add_layer_norm=add_layer_norm,
            pooling_mode=pooling_mode,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        if self.add_cls_pooling_layer:
            sequence_output = outputs[1]
        else:
            # When not using CLS pooling mode, discard it
            sequence_output = outputs[0][:, 1:, :]

        logits = self.pooler(sequence_output, attention_mask)
        logits = self.classifier(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PIXELForQuestionAnswering(ViTForImageClassification):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels

        self.vit = ViTModel(config, add_pooling_layer=False)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = torch.cat([outputs[0][:, 1:, :], outputs[0][:, 0, :].unsqueeze(1)], dim=1)
        # sequence_output = self.dropout(outputs[0][:, 1:, :])

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return x, x


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PIXELPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    Based on timm implementation, which can be found here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class PIXELEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PIXELPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches ** 0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, attention_mask):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # Attention mask indicates patches containing actual text
        # Out of the patches containing actual text we take the one with the highest noise
        # And bump up noise to 100 to guarantee that it gets masked
        # We therefore ensure that at least one masked patch has actual text
        # This is necessary because we only compute loss on patches having text, i.e. loss would otherwise be NaN
        noise_mask = torch.argmax(noise * attention_mask, dim=1)
        noise[torch.arange(noise.size(0)), noise_mask] = 100.0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def controlled_masking(self, sequence, attention_mask, patch_mask):

        batch_size, seq_length, dim = sequence.shape

        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        # We keep the interface the same as in the original random_masking function above
        # The only difference is that instead of random noise we use the predefined mask
        # Sometimes the greedy span masking yields fewer masked patches than specified through mask_ratio
        # We additionally mask out the difference between them randomly using noise in [0, 0.01]
        noise = patch_mask + (torch.rand(batch_size, seq_length, device=sequence.device) / 100)  # noise in [0, 0.01)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def forward(self, pixel_values, attention_mask=None, patch_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        if patch_mask is not None:
            embeddings, attention_mask, mask, ids_restore = self.controlled_masking(
                embeddings, attention_mask, patch_mask
            )
        else:
            embeddings, attention_mask, mask, ids_restore = self.random_masking(embeddings, attention_mask)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        return embeddings, attention_mask, mask, ids_restore


class PIXELSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PIXELModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PIXELSelfOutput(nn.Module):
    """
    The residual connection is defined in PIXELLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class PIXELAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PIXELSelfAttention(config)
        self.output = PIXELSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PIXELIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class PIXELOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class PIXELLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PIXELAttention(config)
        self.intermediate = PIXELIntermediate(config)
        self.output = PIXELOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in PIXEL, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in PIXEL, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output


class PIXELEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PIXELLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PIXELPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PIXELConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PIXELEncoder):
            module.gradient_checkpointing = value


class PIXELModel(PIXELPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = PIXELEmbeddings(config)
        self.encoder = PIXELEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if attention_mask is None:
            attention_mask = torch.ones((pixel_values.shape[0], self.embeddings.num_patches), device=self.device)

        embedding_output, attention_mask, mask, ids_restore = self.embeddings(pixel_values, attention_mask, patch_mask)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, embedding_output.shape, self.device
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return PIXELModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PIXELDecoder(nn.Module):
    def __init__(self, config, num_patches, dtype):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [PIXELLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size ** 2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.dtype = dtype
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches ** 0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        hidden_states,
        ids_restore,
        attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        batch_size = hidden_states.shape[0]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, hidden_states.shape[1] - 1), device=hidden_states.device)

        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, hidden_states.shape, attention_mask.device
        )

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return PIXELDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PIXELForPreTraining(PIXELPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = PIXELModel(config)
        self.decoder = PIXELDecoder(config, num_patches=self.vit.embeddings.num_patches, dtype=self.vit.dtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, attention_mask)  # [N, L, p*p*3]
        logits = decoder_outputs.logits

        merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
        loss = self.forward_loss(pixel_values, logits, merged_mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PIXELForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            attention_mask=attention_mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class PIXELForPreTrainingOutput(ModelOutput):
    """
    Class for PIXELForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are attended to and which are not.
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    attention_mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PIXELModelOutput(ModelOutput):
    """
    Class for PIXELModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
                        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
                    `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PIXELDecoderOutput(ModelOutput):
    """
    Class for PIXELDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None




#####################
#   My classes :)   #
#####################


# class PIXELForImageDiacritizer(PIXELForPreTraining):
class PIXELForIntermediatePreTraining(PIXELForPreTraining):
    """
    Class for intermediate pretraining. In the pretraining, the objective is MLM task,
    where the model is training on restoring the exact masked patches of the input images
    of text.
    There are two differences between the pretraining and the intermediate pretraining:
        a. The model suppose to restore images that are not the same as the input images.
           The inputs are images of undiacritized text, and the outputs are images of the
           same text with diacritization.
        b. The loss is computed on all the text patches. This is because the model should
           predict the diacritics of the words that given to it.
    """

    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        pixel_values=None,
        diacritized_examples=None,
        attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, attention_mask)  # [N, L, p*p*3]
        logits = decoder_outputs.logits

        # from PIL import Image
        # import torchvision.transforms as T
        # transformation = T.ToPILImage()
        # # img0 = transformation(imgs[0])
        # # img0 = transformation(imgs[0:1])
        # pred0 = transformation(logits[0])
        # # img0 = Image.fromarray(imgs[0:1].cpu().detach().numpy())
        # # pred0 = Image.fromarray(pred[0:1].cpu().detach().numpy())
        # # img0.save("diacritics/diacritizing/outputs/img0.png")
        # pred0.save("diacritics/diacritizing/outputs/pred0.png")
        # exit(0)

        # merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
        # changed to apply the loss computation on all the text patches
        merged_mask = torch.bitwise_and(attention_mask == 1, attention_mask == 1).long()
        # the decoder should predict the diacritics
        loss = self.forward_loss(diacritized_examples, logits, merged_mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PIXELForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            attention_mask=attention_mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PIXELBasedDiacritizer(nn.Module):
    def __init__(
            self,
            raw_model_name_or_path,
            candidates_model_name_or_path,
            raw_config,
            candidates_config,
            config_kwargs,
            renderer,
            num_candidates,
            candidates_creator,
            forward_fn=None,
            scoring=lambda t1, t2: torch.einsum('abcd,abcd->abc', t1, t2),
            loss_fn=CrossEntropyLoss()
        ):
        super().__init__()
        
        # input level
        self.num_candidates = num_candidates
        self.candidates_creator = candidates_creator
        self.renderer = renderer
        
        # embedding level
        raw_config.mask_ratio = 0.0
        candidates_config.mask_ratio = 0.0
        self.raw_config = raw_config
        self.candidates_config = candidates_config
        self.raw_vit = PIXELForPreTraining.from_pretrained(
            raw_model_name_or_path,
            from_tf=bool(".ckpt" in raw_model_name_or_path),
            config=self.raw_config,
            **config_kwargs,
        ).vit
        self.candidates_vit = PIXELForPreTraining.from_pretrained(
            candidates_model_name_or_path,
            from_tf=bool(".ckpt" in candidates_model_name_or_path),
            config=self.candidates_config,
            **config_kwargs,
        ).vit
        
        # inference level
        self.forward = forward_fn
        self.scoring = scoring
        self.softmax = torch.nn.Softmax(dim=1) # [bs*nw, nc, ne] -> softmax on num_candidates
        self.loss_fn = loss_fn

        # post inference level
        self.is_adaptive_sampler = None
        self.adaptive_sampling_func = None
        
    def move_to_cuda(self):
        if not next(self.parameters()).is_cuda:
            self.to('cuda')
        print("diacritizer is in cuda:", next(self.parameters()).is_cuda)
    
    def train_only_candidates_vit(self, mode=True):
        """ override train, to update only the candidates vit parameters """
        super(PIXELBasedDiacritizer, self).train(mode)
        self.raw_vit.eval()
        self.candidates_vit.train(mode)

    def train_only_scoring_layer(self, mode=True):
        """ override train, to update only the scoring layer parameters """
        super(PIXELBasedDiacritizer, self).train(mode)
        self.raw_vit.eval()
        self.candidates_vit.eval()
        self.scoring.train(mode)
    
    def full_sentence_forward(
        self,
        raw_pixel_values=None,
        candidates_pixel_values=None,
        attention_mask=None,
        labels=None,
        spans_matrices=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        The candidates are full length sentences. The correct candidate for
        each word found in one of the sentence candidates, independently from
        the other correct candidates for other words
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_outputs = self.raw_vit(
            raw_pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            # patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_latent = raw_outputs.last_hidden_state
        # raw_mask = raw_outputs.mask

        # num_candidates = candidates_pixel_values.shape[0] // raw_pixel_values.shape[0]
        candidates_outputs = self.candidates_vit(
            candidates_pixel_values,
            attention_mask=torch.concatenate([attention_mask] * self.num_candidates),
            head_mask=head_mask,
            # patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state
        # candidates_mask = candidates_outputs.mask

        # compute scoring
        raw_latent = raw_latent.unsqueeze(1) # shape - [bs, 1, 530, hs]
        candidates_latent = candidates_latent.view(
            raw_latent.shape[0],
            self.num_candidates,
            candidates_latent.shape[1],
            candidates_latent.shape[2]
        ) # shape - [bs, nc, 530, 192]
        scores = self.scoring(raw_latent, candidates_latent) # shape - [bs, nc, 530]
        sum_over_words_spans = torch.einsum('bij,bkj->bki', spans_matrices, scores)

        probs = self.softmax(sum_over_words_spans)
        
        # compute loss only on words
        labels = labels * spans_num
        probs = torch.einsum('bki,bi->bki', probs, spans_num)
        loss = self.loss_fn(probs, labels)

        return {
            "loss": loss,
            # the next items for evaluation
            "predictions": probs,
            "label_ids": labels,
            "spans_num": spans_num,
        }
    
    # def forward(
    def inner_product_forward(
        self,
        text=None,
        raw_text=None,
        wordss=None,
        spanss=None,
        candidatess_lists=None,
        raw_examples=None,
        raw_num_patches=None,
        raw_attention_mask=None,
        candidates_examples=None,
        candidatess_num_patches=None,
        candidates_attention_mask=None,
        labels=None,
        indicess=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.raw_config.use_return_dict

        with torch.no_grad():
            raw_outputs = self.raw_vit(
                raw_examples,
                attention_mask=raw_attention_mask,
                head_mask=head_mask,
                # patch_mask=patch_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        raw_latent = raw_outputs.last_hidden_state
        # raw_mask = raw_outputs.mask

        candidates_outputs = self.candidates_vit(
            candidates_examples,
            attention_mask=candidates_attention_mask,
            head_mask=head_mask,
            # patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state
        # candidates_mask = candidates_outputs.mask

        # extract the raw embeddings of the words by their spans
        raw_embeddings = torch.zeros(
            raw_latent.shape[0] * len(candidatess_lists[0]),
            raw_latent.shape[1],
            raw_latent.shape[2]
        ).to(candidates_latent.device) # [bs*nw, ne, hs]
        # slice the embeddings for the multiplication phase
        for i, spans in enumerate(spanss):
            for j ,span in enumerate(spans):
                raw_embeddings[i*len(candidatess_lists[0])+j, 2:2+span[1]-span[0], :] = \
                    raw_latent[i, span[0]:span[1], :] # 2 for CLS and blank column

        # compute scoring
        raw_embeddings = raw_embeddings.unsqueeze(1) # reshape - [bs*nw, 1, ne, hs]
        candidates_latent = candidates_latent.view(
            raw_embeddings.shape[0],
            self.num_candidates,
            candidates_latent.shape[1],
            candidates_latent.shape[2]
        ) # reshape - [bs*nw, nc, ne, hs]
        # sum over the inner products for each candidate
        scores = self.scoring(raw_embeddings, candidates_latent) # shape - [bs*nw, nc]
        
        # compute probabilities
        probs = self.softmax(scores)
        
        # compute loss
        labelss = labels.view(labels.shape[0] * labels.shape[1])
        # labels = labelss.view(labelss.shape[0] * labelss.shape[1])
        loss = self.loss_fn(probs, labelss)
        
        # post inference level, update probability mass of words for next samplings
        if self.is_adaptive_sampler:
            self.adaptive_sampling_func.update_counters(wordss, probs, labelss)
        
        return loss, probs, labelss # probs is the logits


class ViT_Contrastive_Model(nn.Module):
    def __init__(
        self,
        vit_model_name_or_path,
        vit_config,
        config_kwargs,
        renderer,
        num_candidates,
        candidates_creator,
    ):
        super().__init__()
        
        vit_config.mask_ratio = 0.0
        self.vit_config = vit_config
        self.vit_model = PIXELForPreTraining.from_pretrained(
            vit_model_name_or_path,
            from_tf=bool(".ckpt" in vit_model_name_or_path),
            config=self.vit_config,
            **config_kwargs,
        ).vit
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.vit_config.hidden_size, self.vit_config.hidden_size), # Input: e_u1 + delta
            nn.ReLU(),
            nn.Linear(self.vit_config.hidden_size, self.vit_config.hidden_size) # Output: adjusted delta
        )
        self.mse_loss = nn.MSELoss()
        self.config_kwargs = config_kwargs
        self.renderer = renderer
        self.num_candidates = num_candidates
        self.candidates_creator = candidates_creator
    
    def forward(
        self,
        diac_words,
        raw_words,
        raw_examples,
        raw_num_patches,
        raw_attention_mask,
        labelss,
        candidates_list,
        candidate_examples,
        candidate_num_patches,
        candidate_attention_masks,
        margin=0.5,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.vit_config.use_return_dict

        # with torch.no_grad():
        raw_outputs = self.vit_model(
            raw_examples,
            attention_mask=raw_attention_mask,
            head_mask=head_mask,
            # patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_latent = raw_outputs.last_hidden_state # shape - [bs*2, ne, hs]

        # compute mean raw embeddings
        raw_embeddings_list = [rl[:raw_num_patches[i]] \
            for i, rl in enumerate(raw_latent)]
        mean_raw_embeddings = torch.stack([bm.mean(dim=0) for bm in raw_embeddings_list]) # shape - [bs*2, hs]
        
        candidate_outputs = self.vit_model(
            candidate_examples,
            attention_mask=candidate_attention_masks,
            head_mask=head_mask,
            # patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidate_outputs.last_hidden_state # shape - [bs*2*nc, ne, hs]

        # compute mean candidate embeddings
        # num_patches = np.concatenate(candidate_num_patches)
        # cands_embeddings_list = [cl[:num_patches[i]] \
        cands_embeddings_list = [cl[:raw_num_patches[i // self.num_candidates]] \
            for i, cl in enumerate(candidates_latent)]
        mean_cands_embeddings = torch.stack([bm.mean(dim=0) for bm in cands_embeddings_list]) # shape - [bs*2*nc, hs]
        
        # compute MSE loss based on the delta of diac_word and undiac_word
        correct_cands_embeddings = mean_cands_embeddings[labelss[:, 0]]
        delta = correct_cands_embeddings - mean_raw_embeddings # shape - [bs*2, hs]
        # swap even and odd indices, clone to avoid modifying in-place
        swapped_delta = torch.zeros(delta.shape).to(delta.device)
        swapped_delta[0::2, :], swapped_delta[1::2, :] = delta.clone()[1::2, :], delta.clone()[0::2, :]
        # apply MLP with relation to the other undiacritized word, to adjust the delta to the specific word
        swapped_mlp_delta = self.mlp(torch.cat([mean_raw_embeddings, swapped_delta], dim=1))
        # add the delta from the original word, to get the diacritized word
        delta_pred = mean_raw_embeddings + swapped_mlp_delta
        # mean squared error between the shift vectors
        mse_loss = self.mse_loss(correct_cands_embeddings, delta_pred)

        # compute contrastive loss using all the candidates, and first normalize the embeddings
        normalized_raw_embeddings = F.normalize(mean_raw_embeddings, p=2, dim=1)
        normalized_cands_embeddings = F.normalize(mean_cands_embeddings, p=2, dim=1)
        normalized_correct_embeddings = normalized_cands_embeddings[(labelss[:, 0]) % 2]
        normalized_incorrect_embeddings = normalized_cands_embeddings[(labelss[:, 0]+1) % 2]
        # cosine similarity between diacritized, undiacritized and the raw words, to push them apart by a triangle of relationships
        correct_sim = F.cosine_similarity(normalized_raw_embeddings, normalized_correct_embeddings, dim=-1)  # shape - (batch_size,)
        incorrect_sim = F.cosine_similarity(normalized_raw_embeddings, normalized_incorrect_embeddings, dim=-1)  # shape - (batch_size,)
        correct_incorrect_sim = F.cosine_similarity(normalized_correct_embeddings, normalized_incorrect_embeddings, dim=-1)  # shape - (batch_size,)
        # minimize similarity (maximize distance) beyond margin
        correct_loss = torch.relu(correct_sim - (1 - margin))
        incorrect_loss = torch.relu(incorrect_sim - (1 - margin))
        correct_incorrect_loss = torch.relu(correct_incorrect_sim - (1 - margin))
        # contrastive_loss = (correct_loss + incorrect_loss + correct_incorrect_loss).sum()
        contrastive_loss = correct_loss.sum() + incorrect_loss.sum() + correct_incorrect_loss.sum()

        # delta norm loss
        min_delta_norm = 4.0
        delta_norm = torch.norm(delta, p=2, dim=-1)
        delta_norm_loss = F.relu(min_delta_norm - delta_norm).sum()

        # combine losses
        loss = 2 * mse_loss + contrastive_loss / mean_raw_embeddings.shape[0] + delta_norm_loss

        # print(f"mse_loss: {mse_loss * 2}")
        # print(f"contrastive_loss: {contrastive_loss / mean_raw_embeddings.shape[0]}")
        # print(f"delta_norm_loss: {delta_norm_loss}")
        # print(f"delta_norm: {delta_norm}")

        return {'loss': loss}


from transformers import AutoTokenizer, AutoModelForMaskedLM

class Multi_modal_Diacritizer(nn.Module):
    def __init__(
            self,
            raw_model_name_or_path,
            candidates_model_name_or_path,
            candidates_config,
            config_kwargs,
            renderer,
            num_candidates,
            candidates_creator,
            scoring=lambda t1, t2: torch.einsum('ac,abc->ab', t1, t2),
            loss_fn=CrossEntropyLoss()
        ):
        super().__init__()
        
        # input level
        self.num_candidates = num_candidates
        self.candidates_creator = candidates_creator
        self.renderer = renderer
        
        # embedding level
        self.tokenizer = AutoTokenizer.from_pretrained(raw_model_name_or_path)
        self.raw_model = AutoModelForMaskedLM.from_pretrained(raw_model_name_or_path, output_hidden_states=True)
        candidates_config.mask_ratio = 0.0
        self.candidates_config = candidates_config
        
        if not candidates_model_name_or_path:
            self.candidates_vit = PIXELModel(self.candidates_config) # ViT encoder
        else:
            self.candidates_vit = PIXELForPreTraining.from_pretrained(
                candidates_model_name_or_path,
                from_tf=bool(".ckpt" in candidates_model_name_or_path),
                config=self.candidates_config,
                **config_kwargs,
            ).vit
        
        # inference level
        chs = self.raw_model.config.hidden_size
        rhs = self.candidates_vit.config.hidden_size
        self.cands_to_raw_projection = nn.Linear(rhs, chs)
        self.scoring = scoring
        self.softmax = torch.nn.Softmax(dim=1) # [bs*nw, nc, ne] -> softmax on num_candidates
        self.loss_fn = loss_fn

        # post inference level
        self.is_adaptive_sampler = None
        self.adaptive_sampling_func = None
        
        # for the different accuracy metrics
        self.total_char = -1
        self.correct_char = -1
        self.total_words = 0
        self.correct_words = 0
        self.total_voc = 0
        self.correct_voc = 0
        self.total_dec = 0
        self.correct_dec = 0
        # to track after the mistake of the model
        self.true_preds = {}
        self.false_preds = {}

    def move_to_cuda(self):
        if not next(self.parameters()).is_cuda:
            self.to('cuda')
        print("diacritizer is in cuda:", next(self.parameters()).is_cuda)
    
    def train_only_candidates_vit(self, mode=True):
        """ override train, to update only the candidates vit parameters """
        super(Multi_modal_Diacritizer, self).train(mode)
        self.raw_model.eval()
        self.candidates_vit.train(mode)

    def train_only_scoring_layer(self, mode=True):
        """ override train, to update only the scoring layer parameters """
        super(PIXELBasedDiacritizer, self).train(mode)
        self.raw_model.eval()
        self.candidates_vit.eval()
        self.scoring.train(mode)
    
    @staticmethod
    def compute_word_mean_embeddings(embeddings, first_token_mask, input_ids):
        """ Extract the mean embedding for each word """
        sep_tokens_indices = (input_ids == 2).nonzero()[:, 1]
        mean_embeddings = []
        for i in range(embeddings.shape[0]):
            word_embeddings = []
            # Determine the range of indices of each word
            first_token_indices = torch.where(first_token_mask[i])[0]
            for j in range(first_token_indices.shape[0]):
                start_idx = first_token_indices[j]
                end_idx = first_token_indices[j+1] if j + 1 < first_token_indices.shape[0] else sep_tokens_indices[i]
                # Extract the embeddings for the current word
                cur_word_embeddings = embeddings[i, start_idx:end_idx]
                word_embeddings.append(cur_word_embeddings.mean(dim=0))
            mean_embeddings.append(torch.stack(word_embeddings))
        return mean_embeddings
    
    def forward(
        self,
        text=None,
        raw_text=None,
        wordss=None,
        spanss=None,
        candidatess_lists=None,
        raw_examples=None,
        raw_num_patches=None,
        raw_attention_mask=None,
        candidates_examples=None,
        candidatess_num_patches=None,
        candidates_attention_mask=None,
        labels=None,
        indicess=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.candidates_config.use_return_dict

        # Tokenize the sentence and extract mean embeddings of the words
        inputs = self.tokenizer(raw_text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
        inputs.to('cuda')
        offset_mapping = inputs.pop("offset_mapping")

        # with torch.no_grad():
        raw_outputs = self.raw_model(**inputs)
        token_embeddings = raw_outputs.hidden_states[-1] # shape - [bs, nt, rhs]

        # Find the first token of each word
        input_ids = inputs["input_ids"]
        first_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        offset_mapping[:, 0, 1] = -1
        for i, om in enumerate(offset_mapping):
            prev_end = 0
            for idx, (start, end) in enumerate(om.tolist()[:-1]):
                if start != prev_end and end != 0:  # New word starts when current start != previous end
                    first_token_mask[i][idx] = True
                prev_end = end
        
        mean_embeddings = self.compute_word_mean_embeddings(token_embeddings, first_token_mask, input_ids)
        raw_latent = [mean_embeddings[i][indices] for i, indices in enumerate(indicess)]

        # choose the embeddings of the training words
        raw_embeddings = torch.stack(raw_latent).reshape(
            token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        ) # shape - [bs*nw, rhs]
        
        candidates_outputs = self.candidates_vit(
            candidates_examples,
            attention_mask=candidates_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state

        # compute mean candidate embeddings
        num_patches = np.concatenate(candidatess_num_patches)
        cands_embeddings_list = [cl[:num_patches[i // self.num_candidates]] \
            for i, cl in enumerate(candidates_latent)]
        mean_cands_embeddings = torch.stack([bm.mean(dim=0) for bm in cands_embeddings_list]) # shape - [bs*nw*nc, chs]
        
        # project the candidate embeddings from chs to rhs
        proj_cands_embeddings = self.cands_to_raw_projection(mean_cands_embeddings) # shape - [bs*nw*nc, rhs]

        # raw_embeddings = raw_embeddings.unsqueeze(1) # reshape - [bs*nw, 1, ne, hs]
        cands_embeddings = proj_cands_embeddings.view(
            raw_embeddings.shape[0],
            self.num_candidates,
            raw_embeddings.shape[1]
        ) # reshape - [bs*nw, nc, rhs]
        
        # sum over the inner products for each candidate
        scores = self.scoring(raw_embeddings, cands_embeddings) # shape - [bs*nw, nc]
        
        # compute probabilities
        probs = self.softmax(scores)
        
        # compute loss
        labelss = labels.view(labels.shape[0] * labels.shape[1])
        # labels = labelss.view(labelss.shape[0] * labelss.shape[1])
        loss = self.loss_fn(probs, labelss)
        
        # post inference level, update probability mass of words for next samplings
        if self.is_adaptive_sampler:
            self.adaptive_sampling_func.update_counters(wordss, probs, labelss)
        
        # for test phase
        if self.total_char > -1:
            # clean all none non-Hebrew characters and non-Hebrew diacritics
            new_reg = r"[\u05B0-\u05BC\u05C1\u05C2\u05C7]"
            pattern = re.compile(f"(?:{new_reg})|(?:{HEBREW_LETTER})")
            for label, pred, cands in zip(labels[:, 0], probs.argmax(dim=1), candidatess_lists):
                prediction = cands[0][pred]
                prediction = ''.join(c for c in prediction if pattern.fullmatch(c))
                gold = cands[0][label]
                gold = ''.join(c for c in gold if pattern.fullmatch(c))
                pred_letters = split_into_letters_with_diacritics(prediction)
                gold_letters = split_into_letters_with_diacritics(gold)
                if len(pred_letters) != len(gold_letters):
                    raise ValueError("Prediction and gold word must have the same number of base letters.")
                
                # char level accuracy
                cha_correct, cha_total = compute_cha(pred_letters, gold_letters)
                self.correct_char += cha_correct
                self.total_char += cha_total
                # vocalize accuracy
                voc_correct, voc_total = compute_voc(pred_letters, gold_letters)
                self.total_voc += voc_total
                self.correct_voc += voc_correct
                # decisions accuracy
                dec_correct, dec_total = compute_dec(pred_letters, gold_letters)
                self.total_dec += dec_total
                self.correct_dec += dec_correct
            # word level accuracy
            preds = probs.argmax(1)
            for idx, word in enumerate(wordss):
                self.total_words += 1
                word0 = word[0]
                word = normalize_hebrew_text(word[0])
                normalized_cands = [normalize_hebrew_text(cand) for cand in candidatess_lists[idx][0]]
                label = normalized_cands.index(word) if word in normalized_cands else -1
                if label == preds[idx]:
                    self.correct_words += 1
                # track the success and failure of the model
                key = ''.join(c for c in word0 if pattern.fullmatch(c))
                if label == preds[idx]:
                    self.true_preds[key] = self.true_preds.get(key, 0) + 1
                else:
                    self.false_preds[key] = self.false_preds.get(key, 0) + 1

        return loss, probs, labelss # probs is the logits


import os

class CLIP_Diacritizer(nn.Module):
    def __init__(
            self,
            raw_model_name_or_path,
            candidates_model_name_or_path,
            candidates_config,
            config_kwargs,
            renderer,
            num_candidates,
            scoring=lambda t1, t2: torch.einsum('ac,abc->ab', t1, t2),
            loss_fn=CrossEntropyLoss()
        ):
        super().__init__()
        
        # input level
        self.num_candidates = num_candidates
        self.renderer = renderer
        
        # embedding level
        self.tokenizer = AutoTokenizer.from_pretrained(raw_model_name_or_path)
        self.raw_model = AutoModelForMaskedLM.from_pretrained(raw_model_name_or_path, output_hidden_states=True)
        candidates_config.mask_ratio = 0.0
        self.candidates_config = candidates_config
        vit_contrastive_model = ViT_Contrastive_Model(
            vit_model_name_or_path=candidates_model_name_or_path,
            vit_config=self.candidates_config,
            config_kwargs=config_kwargs,
            renderer=self.renderer,
            num_candidates=num_candidates,
            candidates_creator=None,
        )
        vit_contrastive_model.load_state_dict(torch.load(os.path.join(candidates_model_name_or_path, "pytorch_model.bin")))
        self.candidates_vit = vit_contrastive_model.vit_model
        
        # inference level
        rhs = self.raw_model.config.hidden_size
        chs = self.candidates_vit.config.hidden_size
        self.project_cands = nn.Linear(chs, rhs)
        self.project_raw = nn.Linear(rhs, rhs)
        self.scoring = scoring
        self.softmax = torch.nn.Softmax(dim=1) # [bs*nw, nc, ne] -> softmax on num_candidates
        self.loss_fn = loss_fn

        # post inference level
        self.is_adaptive_sampler = None
        self.adaptive_sampling_func = None
        
    def move_to_cuda(self):
        if not next(self.parameters()).is_cuda:
            self.to('cuda')
        print("diacritizer is in cuda:", next(self.parameters()).is_cuda)
    
    def train_only_candidates_vit(self, mode=True):
        """ override train, to update only the candidates vit parameters """
        super(CLIP_Diacritizer, self).train(mode)
        self.raw_model.eval()
        self.candidates_vit.train(mode)

    def train_only_projection_layers(self, mode=True):
        """ override train, to update only the scoring layer parameters """
        super(CLIP_Diacritizer, self).train(mode)
        self.raw_model.eval()
        self.candidates_vit.eval()
        self.project_raw.train(mode)
        self.project_cands.train(mode)
    
    @staticmethod
    def compute_word_mean_embeddings(embeddings, first_token_mask, input_ids):
        """ Extract the mean embedding for each word """
        sep_tokens_indices = (input_ids == 2).nonzero()[:, 1]
        mean_embeddings = []
        for i in range(embeddings.shape[0]):
            word_embeddings = []
            # Determine the range of indices of each word
            first_token_indices = torch.where(first_token_mask[i])[0]
            for j in range(first_token_indices.shape[0]):
                start_idx = first_token_indices[j]
                end_idx = first_token_indices[j+1] if j + 1 < first_token_indices.shape[0] else sep_tokens_indices[i]
                # Extract the embeddings for the current word
                cur_word_embeddings = embeddings[i, start_idx:end_idx]
                word_embeddings.append(cur_word_embeddings.mean(dim=0))
            mean_embeddings.append(torch.stack(word_embeddings))
        return mean_embeddings
    
    def forward(
        self,
        text=None,
        raw_text=None,
        wordss=None,
        spanss=None,
        candidatess_lists=None,
        raw_examples=None,
        raw_num_patches=None,
        raw_attention_mask=None,
        candidates_examples=None,
        candidatess_num_patches=None,
        candidates_attention_mask=None,
        labels=None,
        indicess=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.candidates_config.use_return_dict

        # Tokenize the sentence and extract mean embeddings of the words
        inputs = self.tokenizer(raw_text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
        inputs.to('cuda')
        offset_mapping = inputs.pop("offset_mapping")

        # with torch.no_grad():
        raw_outputs = self.raw_model(**inputs)
        token_embeddings = raw_outputs.hidden_states[-1] # shape - [bs, nt, rhs]

        # Find the first token of each word
        input_ids = inputs["input_ids"]
        first_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        offset_mapping[:, 0, 1] = -1
        for i, om in enumerate(offset_mapping):
            prev_end = 0
            for idx, (start, end) in enumerate(om.tolist()[:-1]):
                if start != prev_end and end != 0:  # New word starts when current start != previous end
                    first_token_mask[i][idx] = True
                prev_end = end
        
        mean_embeddings = self.compute_word_mean_embeddings(token_embeddings, first_token_mask, input_ids)
        raw_latent = [mean_embeddings[i][indices] for i, indices in enumerate(indicess)]

        # choose the embeddings of the training words
        raw_embeddings = torch.stack(raw_latent).reshape(
            token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        ) # shape - [bs*nw, rhs]
        
        candidates_outputs = self.candidates_vit(
            candidates_examples,
            attention_mask=candidates_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state

        # compute mean candidate embeddings
        num_patches = np.concatenate(candidatess_num_patches)
        cands_embeddings_list = [cl[:num_patches[i // self.num_candidates]] \
            for i, cl in enumerate(candidates_latent)]
        mean_cands_embeddings = torch.stack([bm.mean(dim=0) for bm in cands_embeddings_list]) # shape - [bs*nw*nc, chs]
        
        # project the candidate embeddings to a unified orient space
        proj_raw_embeddings = self.project_raw(raw_embeddings)
        proj_cands_embeddings = self.project_cands(mean_cands_embeddings) # shape - [bs*nw*nc, rhs]

        # raw_embeddings = raw_embeddings.unsqueeze(1) # reshape - [bs*nw, 1, ne, hs]
        cands_embeddings = proj_cands_embeddings.view(
            proj_raw_embeddings.shape[0],
            self.num_candidates,
            proj_raw_embeddings.shape[1]
        ) # reshape - [bs*nw, nc, rhs]
        
        raw_embeddings_norm = F.normalize(proj_raw_embeddings, p=2, dim=-1)
        cands_embeddings_norm = F.normalize(cands_embeddings, p=2, dim=-1)
        
        # sum over the inner products for each candidate
        scores = self.scoring(raw_embeddings_norm, cands_embeddings_norm) # shape - [bs*nw, nc]
        # scores = self.scoring(proj_raw_embeddings, cands_embeddings) # shape - [bs*nw, nc]
        
        # compute probabilities
        probs = self.softmax(scores)
        
        # compute loss
        labelss = labels.view(labels.shape[0] * labels.shape[1])
        # labels = labelss.view(labelss.shape[0] * labelss.shape[1])
        loss = self.loss_fn(probs, labelss)
        
        # post inference level, update probability mass of words for next samplings
        if self.is_adaptive_sampler:
            self.adaptive_sampling_func.update_counters(wordss, probs, labelss)
        
        return loss, probs, labelss # probs is the logits


import torch.nn.functional as F
import re

class Reg_Loss_Multi_Modal_Diacritizer(Multi_modal_Diacritizer):
    def __init__(
            self,
            raw_model_name_or_path,
            candidates_model_name_or_path,
            candidates_config,
            config_kwargs,
            renderer,
            num_candidates,
            candidates_creator,
            cands_to_diacritics_projection,
            diacritics,
            scoring=lambda t1, t2: torch.einsum('ac,abc->ab', t1, t2),
            loss_fn=CrossEntropyLoss(),
            lambda_reg=0.5
        ):
        super().__init__(
            raw_model_name_or_path,
            candidates_model_name_or_path,
            candidates_config,
            config_kwargs,
            renderer,
            num_candidates,
            candidates_creator,
            scoring=lambda t1, t2: torch.einsum('ac,abc->ab', t1, t2),
            loss_fn=CrossEntropyLoss()
        )
        self.diacritics = diacritics
        self.diacritics_to_index = {d: i for i, d in enumerate(self.diacritics)}
        self.num_of_diacritics = len(self.diacritics)
        chs = self.candidates_vit.config.hidden_size
        self.cands_to_diacritics_projection = cands_to_diacritics_projection
        # cands_to_diacritics_projection = nn.Linear(chs, self.num_of_diacritics)
        # self.diacritics_loss = nn.BCELoss()
        self.diacritics_loss = nn.BCEWithLogitsLoss()
        self.lambda_reg = lambda_reg  # regularization weight

        # for the different accuracy metrics
        self.total_char = -1
        self.correct_char = -1
        self.total_words = 0
        self.correct_words = 0
        self.total_voc = 0
        self.correct_voc = 0
        self.total_dec = 0
        self.correct_dec = 0
    
    def forward(
        self,
        text=None,
        raw_text=None,
        wordss=None,
        spanss=None,
        candidatess_lists=None,
        raw_examples=None,
        raw_num_patches=None,
        raw_attention_mask=None,
        candidates_examples=None,
        candidatess_num_patches=None,
        candidates_attention_mask=None,
        labels=None,
        indicess=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.candidates_config.use_return_dict

        # Tokenize the sentence and extract mean embeddings of the words
        inputs = self.tokenizer(raw_text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
        inputs.to('cuda')
        offset_mapping = inputs.pop("offset_mapping")

        # with torch.no_grad():
        raw_outputs = self.raw_model(**inputs)
        token_embeddings = raw_outputs.hidden_states[-1] # shape - [bs, nt, rhs]

        # Find the first token of each word
        input_ids = inputs["input_ids"]
        first_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        offset_mapping[:, 0, 1] = -1
        for i, om in enumerate(offset_mapping):
            prev_end = 0
            for idx, (start, end) in enumerate(om.tolist()[:-1]):
                if start != prev_end and end != 0:  # New word starts when current start != previous end
                    first_token_mask[i][idx] = True
                prev_end = end
        
        mean_embeddings = self.compute_word_mean_embeddings(token_embeddings, first_token_mask, input_ids)
        raw_latent = [mean_embeddings[i][indices] for i, indices in enumerate(indicess)]

        # # choose the embeddings of the training words
        # raw_embeddings = torch.stack(raw_latent).reshape(
        #     token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        # ) # shape - [bs*nw, rhs]
        unnormalized_raw_embeddings = torch.stack(raw_latent).reshape(
            token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        ) # shape - [bs*nw, rhs]
        raw_embeddings = F.normalize(unnormalized_raw_embeddings, p=2, dim=1)
        
        candidates_outputs = self.candidates_vit(
            candidates_examples,
            attention_mask=candidates_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state

        # compute mean candidate embeddings
        num_patches = np.concatenate(candidatess_num_patches)
        cands_embeddings_list = [cl[:num_patches[i // self.num_candidates]] \
            for i, cl in enumerate(candidates_latent)]
        mean_cands_embeddings = torch.stack([bm.mean(dim=0) for bm in cands_embeddings_list]) # shape - [bs*nw*nc, chs]
        
        # project the candidate embeddings from chs to rhs
        proj_cands_embeddings = self.cands_to_raw_projection(mean_cands_embeddings) # shape - [bs*nw*nc, rhs]

        # raw_embeddings = raw_embeddings.unsqueeze(1) # reshape - [bs*nw, 1, ne, hs]
        unnormalized_cands_embeddings = proj_cands_embeddings.view(
            raw_embeddings.shape[0],
            self.num_candidates,
            raw_embeddings.shape[1]
        ) # reshape - [bs*nw, nc, rhs]
        cands_embeddings = F.normalize(unnormalized_cands_embeddings, p=2, dim=2)
        
        # sum over the inner products for each candidate
        scores = self.scoring(raw_embeddings, cands_embeddings) # shape - [bs*nw, nc]
        
        # compute probabilities
        probs = self.softmax(scores)
        
        # compute contrastive loss
        labelss = labels.view(labels.shape[0] * labels.shape[1])
        # labels = labelss.view(labelss.shape[0] * labelss.shape[1])
        contrastive_loss = self.loss_fn(probs, labelss)
        
        # compute diacritics loss
        diacritics_labels = self.diacritic_one_hot_vectors(candidatess_lists)
        diacritics_labels = diacritics_labels.to('cuda')
        projected_cands = self.cands_to_diacritics_projection(mean_cands_embeddings)
        diacritic_loss = self.diacritics_loss(projected_cands, diacritics_labels)
        reg_diacritic_loss = diacritic_loss / self.num_candidates
        
        # total loss
        loss = contrastive_loss + self.lambda_reg * reg_diacritic_loss
        
        # post inference level, update probability mass of words for next samplings
        if self.is_adaptive_sampler:
            self.adaptive_sampling_func.update_counters(wordss, probs, labelss)
        
        # for test phase
        if self.total_char > -1:
            # clean all none non-Hebrew characters and non-Hebrew diacritics
            new_reg = r"[\u05B0-\u05BC\u05C1\u05C2\u05C7]"
            pattern = re.compile(f"(?:{new_reg})|(?:{HEBREW_LETTER})")
            for label, pred, cands in zip(labels[:, 0], probs.argmax(dim=1), candidatess_lists):
                prediction = cands[0][pred]
                prediction = ''.join(c for c in prediction if pattern.fullmatch(c))
                gold = cands[0][label]
                gold = ''.join(c for c in gold if pattern.fullmatch(c))
                pred_letters = split_into_letters_with_diacritics(prediction)
                gold_letters = split_into_letters_with_diacritics(gold)
                if len(pred_letters) != len(gold_letters):
                    raise ValueError("Prediction and gold word must have the same number of base letters.")
                
                # char level accuracy
                cha_correct, cha_total = compute_cha(pred_letters, gold_letters)
                self.correct_char += cha_correct
                self.total_char += cha_total
                # vocalize accuracy
                voc_correct, voc_total = compute_voc(pred_letters, gold_letters)
                self.total_voc += voc_total
                self.correct_voc += voc_correct
                # decisions accuracy
                dec_correct, dec_total = compute_dec(pred_letters, gold_letters)
                self.total_dec += dec_total
                self.correct_dec += dec_correct
            # word level accuracy
            preds = probs.argmax(1)
            for idx, word in enumerate(wordss):
                self.total_words += 1
                word = normalize_hebrew_text(word[0])
                normalized_cands = [normalize_hebrew_text(cand) for cand in candidatess_lists[idx][0]]
                label = normalized_cands.index(word) if word in normalized_cands else -1
                if label == preds[idx]:
                    self.correct_words += 1
        
        return loss, probs, labelss # probs is the logits

    def diacritic_one_hot_vectors(self, candidates_lists):
        """
        Converts a Hebrew word with diacritics into a one-hot vector indicating which
        diacritics are present.
        """
        candidates = sum(sum(candidates_lists, []), [])
        one_hot_vectors = torch.zeros((len(candidates), self.num_of_diacritics), dtype=torch.torch.float)
        for i, candidate in enumerate(candidates):
            found_diacritics = re.findall(r'[\u05B0-\u05BC\u05C1\u05C2\u05C7]', candidate)
            for d in found_diacritics:
                if d in self.diacritics_to_index:
                    one_hot_vectors[i, self.diacritics_to_index[d]] = 1
        return one_hot_vectors
    
    def pos_encoding_forward(
        self,
        text=None,
        raw_text=None,
        wordss=None,
        spanss=None,
        candidatess_lists=None,
        raw_examples=None,
        raw_num_patches=None,
        raw_attention_mask=None,
        candidates_examples=None,
        candidatess_num_patches=None,
        candidates_attention_mask=None,
        labels=None,
        indicess=None,
        spans_num=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.candidates_config.use_return_dict

        # Tokenize the sentence and extract mean embeddings of the words
        inputs = self.tokenizer(raw_text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
        inputs.to('cuda')
        offset_mapping = inputs.pop("offset_mapping")

        # with torch.no_grad():
        raw_outputs = self.raw_model(**inputs)
        token_embeddings = raw_outputs.hidden_states[-1] # shape - [bs, nt, rhs]

        # Find the first token of each word
        input_ids = inputs["input_ids"]
        first_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        offset_mapping[:, 0, 1] = -1
        for i, om in enumerate(offset_mapping):
            prev_end = 0
            for idx, (start, end) in enumerate(om.tolist()[:-1]):
                if start != prev_end and end != 0:  # New word starts when current start != previous end
                    first_token_mask[i][idx] = True
                prev_end = end
        
        mean_embeddings = self.compute_word_mean_embeddings(token_embeddings, first_token_mask, input_ids)
        raw_latent = [mean_embeddings[i][indices] for i, indices in enumerate(indicess)]

        # # choose the embeddings of the training words
        # raw_embeddings = torch.stack(raw_latent).reshape(
        #     token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        # ) # shape - [bs*nw, rhs]
        unnormalized_raw_embeddings = torch.stack(raw_latent).reshape(
            token_embeddings.shape[0] * len(wordss[0]), token_embeddings.shape[2]
        ) # shape - [bs*nw, rhs]
        raw_embeddings = F.normalize(unnormalized_raw_embeddings, p=2, dim=1)
        
        candidates_outputs = self.candidates_vit(
            candidates_examples,
            attention_mask=candidates_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        candidates_latent = candidates_outputs.last_hidden_state
        
        cls_embeddings = candidates_latent[:, 0, :]
        no_cls_candidates_latent = candidates_latent[:, 1:, :]

        # compute mean candidate embeddings
        num_patches = np.concatenate(candidatess_num_patches)
        cands_embeddings_list = [cl[:num_patches[i // self.num_candidates]] \
            for i, cl in enumerate(no_cls_candidates_latent)]
        mean_cands_embeddings = torch.stack([bm.mean(dim=0) for bm in cands_embeddings_list]) # shape - [bs*nw*nc, chs]
        
        # project the candidate embeddings from chs to rhs
        proj_cands_embeddings = self.cands_to_raw_projection(mean_cands_embeddings) # shape - [bs*nw*nc, rhs]

        # raw_embeddings = raw_embeddings.unsqueeze(1) # reshape - [bs*nw, 1, ne, hs]
        unnormalized_cands_embeddings = proj_cands_embeddings.view(
            raw_embeddings.shape[0],
            self.num_candidates,
            raw_embeddings.shape[1]
        ) # reshape - [bs*nw, nc, rhs]
        cands_embeddings = F.normalize(unnormalized_cands_embeddings, p=2, dim=2)
        
        # sum over the inner products for each candidate
        scores = self.scoring(raw_embeddings, cands_embeddings) # shape - [bs*nw, nc]
        
        # compute probabilities
        probs = self.softmax(scores)
        
        # compute contrastive loss
        labelss = labels.view(labels.shape[0] * labels.shape[1])
        # labels = labelss.view(labelss.shape[0] * labelss.shape[1])
        contrastive_loss = self.loss_fn(probs, labelss)
        
        # compute diacritics loss
        diacritics_labels = self.diacritic_pos_encoding_handler(candidatess_lists)
        diacritics_labels = diacritics_labels.to('cuda')
        projected_cands = self.cands_to_diacritics_projection(cls_embeddings)
        diacritic_loss = self.diacritics_loss(projected_cands, diacritics_labels)
        reg_diacritic_loss = diacritic_loss / self.num_candidates
        
        # total loss
        loss = contrastive_loss + self.lambda_reg * reg_diacritic_loss
        
        # post inference level, update probability mass of words for next samplings
        if self.is_adaptive_sampler:
            self.adaptive_sampling_func.update_counters(wordss, probs, labelss)
        
        # for test phase
        if self.total_char > -1:
            # clean all none non-Hebrew characters and non-Hebrew diacritics
            new_reg = r"[\u05B0-\u05BC\u05C1\u05C2\u05C7]"
            pattern = re.compile(f"(?:{new_reg})|(?:{HEBREW_LETTER})")
            for label, pred, cands in zip(labels[:, 0], probs.argmax(dim=1), candidatess_lists):
                prediction = cands[0][pred]
                prediction = ''.join(c for c in prediction if pattern.fullmatch(c))
                gold = cands[0][label]
                gold = ''.join(c for c in gold if pattern.fullmatch(c))
                pred_letters = split_into_letters_with_diacritics(prediction)
                gold_letters = split_into_letters_with_diacritics(gold)
                if len(pred_letters) != len(gold_letters):
                    raise ValueError("Prediction and gold word must have the same number of base letters.")
                
                # char level accuracy
                cha_correct, cha_total = compute_cha(pred_letters, gold_letters)
                self.correct_char += cha_correct
                self.total_char += cha_total
                # vocalize accuracy
                voc_correct, voc_total = compute_voc(pred_letters, gold_letters)
                self.total_voc += voc_total
                self.correct_voc += voc_correct
                # decisions accuracy
                dec_correct, dec_total = compute_dec(pred_letters, gold_letters)
                self.total_dec += dec_total
                self.correct_dec += dec_correct
            # word level accuracy
            preds = probs.argmax(1)
            for idx, word in enumerate(wordss):
                self.total_words += 1
                word = normalize_hebrew_text(word[0])
                normalized_cands = [normalize_hebrew_text(cand) for cand in candidatess_lists[idx][0]]
                label = normalized_cands.index(word) if word in normalized_cands else -1
                if label == preds[idx]:
                    self.correct_words += 1
        
        return loss, probs, labelss # probs is the logits
    
    @staticmethod
    def get_positional_encoding(index: int, dim: int = 2, max_len: int = 20) -> torch.Tensor:
        """ Sinusoidal positional encoding function """
        pe = torch.zeros(dim)
        position = index
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(max_len) / dim))
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        # return pe # return values in [-1, 1]
        return pe / 2 + 0.5 # return values in [0, 1]
    
    @staticmethod
    def get_first_diacritic_indices(word: str, diacritics_set: set, max_len) -> dict:
        """ Get first index of each diacritic """
        first_indices = {}
        hebrew_char_index = 0

        for char in word:
            if '\u0590' <= char <= '\u05C7':  # Hebrew diacritics range
                if char in diacritics_set and char not in first_indices:
                    first_indices[char] = hebrew_char_index
            elif hebrew_char_index < max_len:
                hebrew_char_index += 1

        return first_indices
    
    def diacritic_pos_encoding(self, word: str, dim: int = 2, max_len: int = 20) -> torch.Tensor:
        """
        Build an auxiliary vector by converting a Hebrew word with diacritics into a 2D matrix:
        a. one-hot vector indicating which diacritics are present,
        b. and posional encoding of the first index the specific diacritic appears at.
        The returned matrix is flattened into a 1D matrix of concatenation of the rows
        """
        one_hot = torch.zeros(len(self.diacritics))
        pos_encodings = torch.zeros(len(self.diacritics), dim)
        indices = Reg_Loss_Multi_Modal_Diacritizer.get_first_diacritic_indices(word, set(self.diacritics), max_len)
        for i, d in enumerate(self.diacritics):
            if d in word:
                one_hot[i] = 1.0
            if d in indices:
                pos_encodings[i] = Reg_Loss_Multi_Modal_Diacritizer.get_positional_encoding(indices[d], dim, max_len)
        # change the shape into a 1D vector: [num_diacritics, dim + 1] -> [num_diacritics * (dim + 1)]
        return torch.cat([one_hot.unsqueeze(1), pos_encodings], dim=1).view(len(self.diacritics) * 3)
    
    def diacritic_pos_encoding_handler(self, candidates_lists):
        """
        Converts a Hebrew word with diacritics into a one-hot vector indicating which
        diacritics are present.
        """
        candidates = sum(sum(candidates_lists, []), [])
        one_hot_vectors = torch.zeros((len(candidates), self.num_of_diacritics * 3), dtype=torch.torch.float)
        for i, candidate in enumerate(candidates):
            one_hot_vectors[i, :] = self.diacritic_pos_encoding(candidate)
        return one_hot_vectors


# for char level accuracy in the test phase
HEBREW_DIACRITICS = "\u05B0-\u05BC\u05C1\u05C2\u05C7"

# Precompiled regex for efficiency
HEBREW_LETTER = r"[\u05D0-\u05EA]"
DIACRITIC_REGEX = re.compile(f"[{HEBREW_DIACRITICS}]")

def split_into_letters_with_diacritics(word):
    """
    Split a Hebrew word into a list of (base_letter, set_of_diacritics).
    """
    result = []
    current_letter = None
    current_diacritics = []
    
    for char in word:
        if re.match(HEBREW_LETTER, char):
            if current_letter is not None:
                result.append((current_letter, set(current_diacritics)))
            current_letter = char
            current_diacritics = []
        elif re.match(DIACRITIC_REGEX, char):
            if current_letter is not None:
                current_diacritics.append(char)
    if current_letter is not None:
        result.append((current_letter, set(current_diacritics)))
    
    return result

def compute_cha(pred_letters, gold_letters):
    """
    Calculate character-level diacritic accuracy between a predicted word and a gold label.
    
    Args:
        prediction (str): The predicted diacritized word.
        gold (str): The gold standard diacritized word.
    
    Returns:
        accuracy (float): The fraction of correctly predicted characters (diacritics-wise).
    """
    
    correct = 0
    total = len(gold_letters)
    
    for (pred_base, pred_diacritics), (gold_base, gold_diacritics) in zip(pred_letters, gold_letters):
        if pred_base != gold_base:
            continue  # different base letters — considered wrong (optional: raise error)
        if pred_diacritics == gold_diacritics:
            correct += 1
    
    return correct, total

def normalize_diacritics(word):
    """
    Splits a Hebrew diacritized word into a list where each element is a character
    with its diacritics (sorted to ignore order differences).
    """
    HEBREW_LETTERS = "\u05D0-\u05EA"
    HEBREW_DIACS = "\u05B0-\u05BC\u05C1\u05C2\u05C7"  # Includes vowels & shin/sin marks

    pattern = re.compile(f"([{HEBREW_LETTERS}])([{HEBREW_DIACS}]*)")

    # Convert each character to a tuple of (letter, sorted diacritics)
    return "".join(match[1] + "".join(sorted(match[2])) for match in pattern.finditer(word))

def normalize_hebrew_text(text):
    """
    Normalizes Hebrew diacritics in a long string while keeping non-Hebrew content unchanged.
    """
    HEBREW_WORD_PATTERN = re.compile(r"[\u05D0-\u05EA\u05B0-\u05BC\u05C1\u05C2\u05C7]+")  # Detects Hebrew words with diacritics

    return HEBREW_WORD_PATTERN.sub(lambda match: normalize_diacritics(match.group()), text)


import unicodedata

A_VOWELS = {'\u05B7', '\u05B8'}               # patach, kamatz
E_VOWELS = {'\u05B6', '\u05B5'}               # segol, tzere
I_VOWELS = {'\u05B4'}                         # hiriq
O_VOWELS = {'\u05B9'}                         # holam
U_VOWELS = {'\u05BB', '\u05BC'} - {'\u05BC'}  # kubutz, shuruk (U+05BC = dagesh, exclude it)
NULL_VOWELS = set()

SCHWA = '\u05B0'
DAGESH_RELEVANT = {'ב', 'כ', 'פ'}

# All vowels
ALL_VOWELS = A_VOWELS | E_VOWELS | I_VOWELS | O_VOWELS | U_VOWELS

# Diacritic Unicode points
DAGESH = '\u05BC'
SHIN_DOT = '\u05C1'
SIN_DOT = '\u05C2'

def get_vowel_class(diacritics):
    """
    Maps a set of diacritics to a vowel class string: 'a', 'e', 'i', 'o', 'u', or 'null'.
    """
    for d in diacritics:
        if d == SCHWA:
            continue  # ignore schwa
        if d in A_VOWELS:
            return 'a'
        if d in E_VOWELS:
            return 'e'
        if d in I_VOWELS:
            return 'i'
        if d in O_VOWELS:
            return 'o'
        if d in U_VOWELS:
            return 'u'
    return 'null'

def get_shin_type(base, diacritics):
    if base != 'ש':
        return None
    if SHIN_DOT in diacritics:
        return 'shin'
    elif SIN_DOT in diacritics:
        return 'sin'
    else:
        return 'none'

def compute_voc(pred_letters, gold_letters):
    """
    Computes the VOC (vocalization-accuracy) score.
    Args:
        predictions: list of predicted diacritized Hebrew words.
        golds: list of gold-standard diacritized Hebrew words.
    Returns:
        VOC accuracy (float).
    """
    correct = 0
    total = 0

    word_is_correct = True

    for (pred_base, pred_diac), (gold_base, gold_diac) in zip(pred_letters, gold_letters):
        if pred_base != gold_base:
            word_is_correct = False
            break

        # Dagesh
        if pred_base in DAGESH_RELEVANT:
            if has_dagesh(pred_diac) != has_dagesh(gold_diac):
                word_is_correct = False
                break

        # Shin/sin dot
        if pred_base == 'ש':
            if get_shin_type(pred_base, pred_diac) != get_shin_type(gold_base, gold_diac):
                word_is_correct = False
                break

        # Vowel class
        if get_vowel_class(pred_diac) != get_vowel_class(gold_diac):
            word_is_correct = False
            break

    total += 1
    if word_is_correct:
        correct += 1

    return (correct, total) if total > 0 else (0, 0)

# Hebrew vowels range (U+05B0 to U+05BB inclusive)
VOWELS = {chr(cp) for cp in range(0x05B0, 0x05BC)}
# Hebrew letters that can have dagesh
can_dagesh = set('בגדהוזטיכלמנספצקשת' + 'ךף')
# Hebrew letters that can have niqqud
can_niqqud = set('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן')

def split_base_and_diacritics(word):
    """
    Splits a Hebrew word into list of (base_char, [diacritics]) tuples.
    """
    result = []
    base = None
    diacritics = []
    for ch in word:
        if unicodedata.combining(ch):
            diacritics.append(ch)
        else:
            if base is not None:
                result.append((base, diacritics))
            base = ch
            diacritics = []
    if base is not None:
        result.append((base, diacritics))
    return result

def has_dagesh(diacritics):
    return DAGESH in diacritics

def get_shin_dot(diacritics):
    if SHIN_DOT in diacritics:
        return 'shin'
    elif SIN_DOT in diacritics:
        return 'sin'
    else:
        return 'none'

def get_vowel(diacritics):
    for d in diacritics:
        if d in VOWELS:
            return d
    return None

def compute_dec(pred_letters, gold_letters):
    """
    Computes the DEC score for lists of predicted and gold Hebrew words.

    Args:
        predictions: list of predicted diacritized Hebrew words.
        golds: list of gold-standard diacritized Hebrew words.

    Returns:
        DEC accuracy (float).
    """
    correct = 0
    total = 0

    for (pred_base, pred_diac), (gold_base, gold_diac) in zip(pred_letters, gold_letters):
        assert pred_base == gold_base, f"Base char mismatch: {pred_base} vs {gold_base}"

        # Shin dot decision
        if gold_base == 'ש':
            total += 1
            if get_shin_dot(pred_diac) == get_shin_dot(gold_diac):
                correct += 1

        # Dagesh decision
        if gold_base in can_dagesh:
            total += 1
            if has_dagesh(pred_diac) == has_dagesh(gold_diac):
                correct += 1

        # Vowel decision
        if gold_base in can_niqqud:
            total += 1
            pred_vowel = get_vowel(pred_diac)
            gold_vowel = get_vowel(gold_diac)
            if pred_vowel == gold_vowel:
                correct += 1

    return (correct, total) if total > 0 else (0, 0)
