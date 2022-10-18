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


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    bsz = visual_input.size(0)
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model({"image": visual_input, "text_input": text_input}, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(bsz, 12, -1, 24, 24) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(bsz, 12, -1, 24, 24) * mask

        gradcams = cams * grads
        gradcams_list = []

        for ind in range(bsz):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            gradcam = gradcam[1:token_length_ + 1, :].sum(dim=0, keepdim=True) / token_length_
            gradcams_list.append(gradcam)

    return gradcams_list