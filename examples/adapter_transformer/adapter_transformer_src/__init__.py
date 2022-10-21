# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .criterions import (label_smoothed_cross_entropy_encoder_similarity,
                         language_classification_cross_entropy,
                         label_smoothed_cross_entropy_codebook,
                         )
from .models import (adapter_transformer,
                     language_classification_transformer,
                     codebook_transformer,
                     )
from .modules import adapter_transformer_layer, classifier
from .tasks import (language_label_probing,
                    multilingual_translation_adapter,
                    multilingual_translation_with_adversarial_language_classifier,
                    )
