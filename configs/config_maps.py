MODEL_PROTOTYPE_CONFIGS = {
    "scratch_noto_span0.25-dropout": "configs/models/scratch_noto_span0.25-dropout.json"
}

TRAINING_CONFIGS = {
    "bs2": "configs/training/bs2.json",
    "tiny": "configs/training/tiny.json",
    "small": "configs/training/small.json",
    "base": "configs/training/base.json",
    "fp16_apex_bs32": "configs/training/fp16_apex_bs32.json"
}

INTERMEDIATE_CONFIGS = {
    "small": "configs/intermediate/small.json",
    "base": "configs/intermediate/base.json"
}

MODELING_DIACRITICS_CONFIGS = {
    "base": "configs/modeling_diacritics/base.json"
}

FINETUNING_CONFIGS = {
    "small": "configs/finetuning/small.json",
    "base": "configs/finetuning/base.json"
}

MM_CONFIGS = {
    "small": "configs/mm_finetuning/small.json",
    "base": "configs/mm_finetuning/base.json"
}

MM_WORDS_CONFIGS = {
    "small": "configs/mm_words_finetuning/small.json",
    "base": "configs/mm_words_finetuning/base.json"
}

REG_LOSS_CONFIGS = {
    "base": "configs/reg_loss_finetuning/base.json"
}

CLIP_CONFIGS = {
    "base": "configs/clip_finetuning/base.json"
}
