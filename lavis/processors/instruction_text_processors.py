"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random
import nltk
import pickle

TEMPLATES = {"image":{}, "pc":{}, "audio":{}, "video":{}}


TEMPLATES["video"]["qa"] = [
"Given the video, {}",
"Q: {} A:",
"Answer the following question based on the video: {}",
"Question: {} Answer:",
"How would you answer {} after watching the video?",
"What is the answer to the question {} after viewing the video?",
"Answer the question based on the video. Question: {} Answer: ",
"Instruction: Answer the following question by reference to the input video. Question: {} Answer:",
"Given the video, what is the answer to the question {}?",
"What's your response to the query {} after watching the video?",
"Please provide an answer to {} after watching the video",
"Respond to the query {} based on the video",
"Based on the given video, respond to {}",
"Question: {} What's your response after watching the video?",
"Consider the following query: {}",
"Could you help answer the question {}?",
"Referencing the provided video, can you answer the question {}?",
"With respect to the video shown, please answer {}",
"What's your answer to {} in the context of the provided video?",
"Question (refer to the video for context): {} Answer:",
"In response to the question {}, what would your answer be after viewing the video?"
]

TEMPLATES["video"]["classification"] = [
"Identify the objects in this video:",
"What objects can you spot in the clip?",
"Classify the objects in the video:",
"Name the objects visible in the video:",
"Identify and classify the objects in this video:",
"What type of objects are in the video?",
"Classify the main object in the clip:",
"Identify the primary object in this video:",
"Name the main object visible in the clip:",
"What is the central object in the video?",
"Can you classify the central object in this video?",
"Determine the type of the main object in this video:",
"What objects do you identify in the video?",
"Please classify the objects in the video:",
"Please identify the main object in the video:",
"What kind of object is displayed in the video?",
"What object is predominantly featured in the video?",
"Classify the object that is the focus of this video:",
"What is the main object in the video?",
"Identify the primary object in the video:"
]

TEMPLATES["video"]["caption"] = [
"A short caption for the video:",
"A short description of the video:",
"A video of",
"A video that shows",
"Describe the video briefly.",
"Write a description for the video.",
"Provide a description of what is presented in the video.",
"Briefly describe the content of the video.",
"Can you briefly explain what you see in the video?",
"Could you use a few words to describe what you perceive in the video?",
"Please provide a short description of the video.",
"Using language, provide a short account of the video.",
"Use a few words to illustrate what is happening in the video."
]

TEMPLATES["image"]["qa"] = [
    "{}",
    "Q: {} A:",
    "Answer the following question: {}",
    "Question: {} Answer:",
    "How would you answer {}?",
    "What is the answer to the question {}?",
    "Answer the question based on the image. Question: {} Answer: ",
    "Instruction: Answer the following question by reference to the input image. Question: {} Answer:",
    "Given the photo, what is the answer to the question {}?",
    "What's your response to the query {}?",
    "Please provide an answer to {}",
    "Respond to the query {}",
    "Based on the given image, respond to {}",
    "Question: {} What's your response?",
    "Consider the following query: {}",
    "Could you help answer the question {}?",
    "Referencing the provided image, can you answer the question {}?",
    "With respect to the image shown, please answer {}",
    "What's your answer to {} in the context of the provided image?",
    "Question (refer to the image for context): {} Answer:",
    "In response to the question {}, what would your answer be?",
    "Considering the details of the image, {}",
    "Contemplating the image, {}",
    "Upon examining the image, how would you address {}?",
    "Using the image as a guideline, {}",
    "From the perspective of the image, {}",
    "Please provide a detailed answer to {} based on your insights from the image.",
    "If you were to use the image as a sole reference, how would you address {}?",
    "Factoring in the image's details, what's your take on {}?",
    "In light of the provided image, {}",
    "Relying solely on the image, how would you tackle {}?",
    "Referencing the intricate details of the image, {}",
    "Giving heed to the image, how would you reply to {}?",
    "In the context of the image's attributes, {}",
    "Having the image in mind, provide an answer to {}.",
    "Reflecting upon the image, {}",
    "Question: {}. Keeping the image in mind, please provide an answer.",
    "By referencing the key aspects of the image, {}",
    "While considering the intricacies of the image, how would you approach answering {}?",
    "Having thoroughly reviewed the image, please respond to {}.",
    "Let's focus on the image. In its context, how would you interpret {}?",
    "With the nuances of the image at the forefront, {}",
    "Taking a cue from the image, {}",
    "Bearing in mind the characteristics of the image, how would you elucidate {}?",
    "Gauging by the image, {}",
    "In accordance with the image's specifications, {}",
    "Drawing insights from the image, {}"
]

TEMPLATES["image"]["classification"] = [
    "Identify the objects in this image:",
    "What objects can you spot in the picture?",
    "Classify the objects in the photo:",
    "Name the objects visible in the image:",
    "Identify and classify the objects in this image:",
    "What type of objects are in the photo?",
    "Classify the main object in the picture:",
    "Identify the primary object in this image:",
    "Name the main object visible in the photo:",
    "What is the central object in the image?",
    "Can you classify the central object in this photo?",
    "Determine the type of the main object in this picture:",
    "What objects do you identify in the photo?",
    "Please classify the objects in the image:",
    "Please identify the main object in the picture:",
    "What kind of object is displayed in the image?",
    "What object is predominantly featured in the photo?",
    "Classify the object that is the focus of this image:",
    "What is the main object in the picture?",
    "Identify the primary object in the photo:"
]

TEMPLATES["image"]["caption"] = [
    "A short caption:",
    "A short description:",
    "A short image caption:",
    "A short image description:",
    "A photo of",
    "A photo that shows",
    "A picture of",
    "A picture that shows",
    "An image of",
    "An image that shows",
    "Write a short description.",
    "Write a description for the image.",
    "Provide a description of what is presented in the image.",
    "Briefly describe the content of the image.",
    "Can you briefly explain what you see in the image?",
    "Could you use a few words to describe what you perceive in the image?",
    "Please provide a short description of the image.",
    "Using language, provide a short account of the image.",
    "Use a few words to illustrate what is happening in the photo."
    "Write a description for the photo.",
    "Provide a description of what is presented in the photo.",
    "Briefly describe the content of the photo.",
    "Can you briefly explain what you see in the photo?",
    "Could you use a few words to describe what you perceive in the photo?",
    "Please provide a short description of the picture.",
    "Using language, provide a short account of the picture.",
    "Use a few words to illustrate what is happening in the picture.",
    "Write a description for the picture.",
    "Provide a description of what is presented in the picture.",
    "Briefly describe the content of the picture.",
    "Can you briefly explain what you see in the picture?",
    "Could you use a few words to describe what you perceive in the picture?",
    "Please provide a short description of the picture.",
    "Using language, provide a short account of the picture.",
    "Use a few words to illustrate what is happening in the picture.",
    "What can you tell me about this image?",
    "Please detail what's in this image.",
    "Narrate the scene captured in this image.",
    "Describe the main elements in this photo.",
    "Tell me what stands out in this picture.",
    "What is the focal point of this image?",
    "Please identify the primary subjects in this photo.",
    "Detail the atmosphere or mood of this image.",
    "Paint a verbal picture based on this image.",
    "Elaborate on the contents of this photo.",
    "How would you introduce this image to someone?",
    "Share a summary of this image.",
    "What are the notable features of this image?",
    "Highlight the main themes present in this photo.",
    "Sketch the scene using words.",
    "Enumerate the elements you see in this image.",
    "Characterize the essence of this picture.",
    "Relate the story depicted in this image.",
    "Describe this image as if speaking to someone who can't see it."
    ]



TEMPLATES["audio"]["caption"] = [
    "A short caption:",
    "A short description:",
    "An audio of",
    "An audio that shows",
    "Write a short description.",
    "Write a description for the audio.",
    "Provide a description of what is presented in the audio.",
    "Briefly describe the content of the audio.",
    "Can you briefly explain what you hear in the audio?",
    "Could you use a few words to describe what you perceive in the audio?",
    "Please provide a short description of the audio.",
    "Using language, provide a short account of the audio.",
    "Use a few words to illustrate what is happening in the audio.",
    "Describe briefly the contents of the audio.",
    "Please provide a brief summary of the audio.",
    "What does the audio contain?",
    "What can you hear in the audio?",
    "What sounds are present in the audio?",
    "Summarize the audio in a few words.",
    "Write a brief summary of the audio content.",
    "Could you provide a concise explanation of the audio's contents?",
    "Describe what the audio represents.",
    "What is the audio depicting?",
    "In a few words, describe what you hear in the audio."
]


TEMPLATES["audio"]["classification"] = [
    "Classify the following audio:",
    "What is the category of this audio clip?",
    "Identify the content of the following audio:",
    "Provide a classification for the audio.",
    "Analyze and categorize the following audio.",
    "Describe the category of the given audio.",
    "Determine the type of this audio clip.",
    "Can you classify what you hear in the audio?",
    "What type of audio is this?",
    "How would you classify this audio clip?",
    "Please identify the category of the following audio:",
    "What category does the following audio fall into?",
    "Classify the sounds in this audio clip."
]

TEMPLATES["audio"]["qa"] = [
    "{}",
    "Question: {} Answer:",
    "Q: {} A:",
    "Based on the audio, {}",
    "Answer the following question based on the audio: {}",
    "Question: {} Provide an answer based on the audio.",
    "How would you answer {} based on the audio?",
    "What is the answer to the question {} using the audio as a reference?",
    "Answer the question using the audio. Question: {} Answer: ",
    "Instruction: Answer the following question by referencing the audio. Question: {} Answer:",
    "Given the audio, what is the answer to the question {}?",
    "What's your response to the query {} considering the audio?",
    "Please provide an answer to {} using the audio as context.",
    "Respond to the query {} based on the audio content.",
    "Based on the provided audio, respond to {}",
    "Question: {} What's your response using the audio for context?",
    "Consider the following query and the audio: {}",
    "Could you help answer the question {} using the audio as reference?",
    "Referencing the provided audio, can you answer the question {}?",
    "With respect to the audio provided, please answer {}",
    "What's your answer to {} in the context of the provided audio?",
    "Question (refer to the audio for context): {} Answer:",
    "In response to the question {}, what would your answer be based on the audio?",
    "Given the audio, how would you respond to {}?",
    "Taking the audio into consideration, what is your response to {}?",
    "Based on the audio, how would you answer {}?"
]

TEMPLATES["pc"]["caption"] = [
    "A short caption:",
    "A short description:",
    "A short 3D caption:",
    "A short 3D description:",
    "A 3D model of",
    "A 3D model that shows",
    "Write a short description.",
    "Write a description for the 3D model.",
    "Provide a description of what is presented in the 3D model.",
    "Briefly describe the content of the 3D model.",
    "Can you briefly explain what you see in the 3D model?",
    "Could you use a few words to describe what you perceive in the 3D model?",
    "Please provide a short description of the 3D model.",
    "Using language, provide a short account of the 3D model.",
    "Use a few words to illustrate what is happening in the 3D model.",
    "Describe briefly the contents of the 3D model.",
    "Please provide a brief summary of the 3D model.",
    "What does the 3D model contain?",
    "What can you identify in the 3D model?",
    "What structures are present in the 3D model?",
    "Summarize the 3D model in a few words.",
    "Write a brief summary of the 3D model content.",
    "Could you provide a concise explanation of the 3D model's contents?",
    "Describe what the 3D model represents.",
    "What is the 3D model depicting?",
    "In a few words, describe what you see in the 3D model.",
    "What can you tell me about this image?",
    "Please detail what's in this 3D model.",
    "Narrate the scene captured in this 3D model.",
    "Describe the main elements in this 3D model.",
    "Tell me what stands out in this 3D model.",
    "What is the focal point of this 3D model?",
    "Please identify the primary subjects in this 3D model.",
    "Detail the atmosphere or mood of this 3D model.",
    "Paint a verbal picture based on this 3D model.",
    "Elaborate on the contents of this 3D model.",
    "How would you introduce this 3D model to someone?",
    "Share a summary of this 3D model.",
    "What are the notable features of this 3D model?",
    "Highlight the main themes present in this 3D model.",
    "Sketch the scene using words.",
    "Enumerate the elements you see in this 3D model.",
    "Characterize the essence of this 3D model.",
    "Relate the story depicted in this 3D model.",
    "Describe this 3D model as if speaking to someone who can't see it."
]

TEMPLATES["pc"]["classification"] = [
    "Classify the following 3D model:",
    "What is the category of this 3D model?",
    "Identify the content of the following 3D model:",
    "Provide a classification for the 3D model.",
    "Analyze and categorize the following 3D model.",
    "Describe the category of the given 3D model.",
    "Determine the type of this 3D model.",
    "Can you classify what you see in the 3D model?",
    "What type of 3D model is this?",
    "How would you classify this 3D model?",
    "Please identify the category of the following 3D model:",
    "What category does the following 3D model fall into?",
    "Classify the structures in this 3D model."
]

TEMPLATES["pc"]["qa"] = [
    "{}",
    "Question: {} Answer:",
    "Q: {} A:",
    "Based on the 3D model, {}",
    "Answer the following question based on the 3D model: {}",
    "Question: {} Provide an answer based on the 3D model.",
    "How would you answer {} based on the 3D model?",
    "What is the answer to the question {} using the 3D model as a reference?",
    "Answer the question using the 3D model. Question: {} Answer: ",
    "Instruction: Answer the following question by referencing the 3D model. Question: {} Answer:",
    "Given the 3D model, what is the answer to the question {}?",
    "What's your response to the query {} considering the 3D model?",
    "Please provide an answer to {} using the 3D model as context.",
    "Respond to the query {} based on the 3D model content.",
    "Based on the provided 3D model, respond to {}",
    "Question: {} What's your response using the 3D model for context?",
    "Consider the following query and the 3D model: {}",
    "Could you help answer the question {} using the 3D model as reference?",
    "Referencing the provided 3D model, can you answer the question {}?",
    "With respect to the 3D model provided, please answer {}",
    "What's your answer to {} in the context of the provided 3D model?",
    "Question (refer to the 3D model for context): {} Answer:",
    "In response to the question {}, what would your answer be based on the 3D model?",
    "Given the 3D model, how would you respond to {}?",
    "Taking the 3D model into consideration, what is your response to {}?",
    "Based on the 3D model, how would you answer {}?",
    "Considering the details of the 3D model, {}",
    "Contemplating the 3D model, {}",
    "Upon examining the 3D model, how would you address {}?",
    "Using the 3D model as a guideline, {}",
    "From the perspective of the 3D model, {}",
    "Please provide a detailed answer to {} based on your insights from the 3D model.",
    "If you were to use the 3D model as a sole reference, how would you address {}?",
    "Factoring in the 3D model's details, what's your take on {}?",
    "In light of the provided 3D model, {}",
    "Relying solely on the 3D model, how would you tackle {}?",
    "Referencing the intricate details of the 3D model, {}",
    "Giving heed to the 3D model, how would you reply to {}?",
    "In the context of the 3D model's attributes, {}",
    "Having the 3D model in mind, provide an answer to {}.",
    "Reflecting upon the 3D model, {}",
    "Question: {}. Keeping the 3D model in mind, please provide an answer.",
    "By referencing the key aspects of the 3D model, {}",
    "While considering the intricacies of the 3D model, how would you approach answering {}?",
    "Having thoroughly reviewed the 3D model, please respond to {}.",
    "Let's focus on the 3D model. In its context, how would you interpret {}?",
    "With the nuances of the 3D model at the forefront, {}",
    "Taking a cue from the 3D model, {}",
    "Bearing in mind the characteristics of the 3D model, how would you elucidate {}?",
    "Gauging by the 3D model, {}",
    "In accordance with the 3D model's specifications, {}",
    "Drawing insights from the 3D model, {}"
]


@registry.register_processor("blip_instruction")
class BlipInstructionProcessor(BaseProcessor):
    def __init__(self, prompt, max_words, modality, task, cmu_dict_path):
        self.prompt = prompt
        self.max_words = max_words
        self.modality = modality
        self.task = task
        if task == 'classification':
            ## download cmu_dict and save it as pickle file 
            if cmu_dict_path:
                self.pronounciations = pickle.load(open(cmu_dict_path, 'rb'))
            else:
                # try:
                #     nltk.download('cmudict')
                # except:
                #     pass
                from nltk.corpus import cmudict
                self.pronounciations = cmudict.dict()
            
           
    def classification_output(self, label):
        if self.starts_with_vowel_sound(label) or label[0] in 'aeiou':
            label = 'an ' + label
        else:
            label = 'a ' + label
        if self.modality == "audio":
            prompts = ["This is ", "It is ", "I hear ", "The audio is ", "This is the sound of ", ""]
        else:
            prompts = ["", "This is ", "It is "]
        caption = random.choice(prompts) +  label
        return caption.lower()
    
    def starts_with_vowel_sound(self,word):
        for syllables in self.pronounciations.get(word, []):
            return syllables[0][-1].isdigit()  # use only the first one
                

    def __call__(self, caption):
        if self.task == 'eval':
            if caption == "":
                caption = self.prompt
            else:
                caption = self.pre_caption(caption)
            return caption
        if caption == "" and self.task == 'caption':
            caption = self.prompt + random.choice(TEMPLATES[self.modality][self.task])
        elif self.task == 'qa':
            caption = self.prompt + random.choice(TEMPLATES[self.modality][self.task]).format(caption).replace('??', '?')
        elif self.task == 'classification':
            if caption == "":
                caption = self.prompt + random.choice(TEMPLATES[self.modality][self.task])
            else:
                caption = self.classification_output(caption)
        else:
            caption = self.pre_caption(caption)

        return caption.lower().strip()

    @classmethod
    def from_config(classification, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)
        modality = cfg.get("modality", 'image')
        task = cfg.get("task", 'caption')
        cmu_dict_path = cfg.get("cmu_dict_path", None)
        return classification(prompt=prompt, max_words=max_words, modality=modality, task=task, cmu_dict_path=cmu_dict_path)

    def pre_caption(self, caption):
        if isinstance(caption,list):
            caption = random.choice(caption)
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption