"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch


def prepare_qa_input(sample, num_captions, num_captions_fid):
    sample_question_captions = []

    for question, captions in zip(sample['text_input'], sample['captions']):
        assert isinstance(captions, list)
        question_captions = []
        question_caption = ''
        for cap_id, cap_ in enumerate(captions[0:num_captions]):
            question_caption += (cap_.strip() + '. ')
            if (cap_id + 1) != num_captions and ((cap_id + 1) % num_captions_fid == 0):
                question_caption = question.lower().strip() + " \\n " + question_caption.lower().strip()
                question_captions.append(question_caption)
                question_caption = ''
            if (cap_id + 1) == num_captions:
                question_caption = question.lower().strip() + " \\n " + question_caption.lower().strip()
                question_captions.append(question_caption)
        sample_question_captions.append(question_captions)

    sample['question_captions'] = sample_question_captions
