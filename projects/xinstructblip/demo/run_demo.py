"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gradio as gr
from lavis.models.blip2_models.blip2_vicuna_xinstruct import Blip2VicunaXInstruct
import torch
import argparse
import numpy as np
from lavis.processors.ulip_processors import ULIPPCProcessor
from lavis.processors.clip_processors import ClipImageEvalProcessor
from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from omegaconf import OmegaConf
from lavis.common.registry import registry
import random

import trimesh
import pyvista as pv

#https://github.com/mikedh/trimesh/issues/507
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def convert_mesh_to_numpy(mesh_file, npoints=8192):
    print("Loading point cloud.")
    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(mesh_file, force='mesh')
    mesh = as_mesh(mesh)

    # Subsample or upsample the mesh to have exactly npoints points
    vertices = mesh.vertices
    num_points = len(vertices)
    if num_points < npoints:
        # Upsample the mesh by repeating vertices
        repetitions = int(np.ceil(npoints / num_points))
        vertices = np.repeat(vertices, repetitions, axis=0)[:npoints]
    elif num_points > npoints:
        # Subsample the mesh to the desired number of points
        # indices = trimesh.sample.sample_surface(mesh, npoints)[0]
        vertices = mesh.vertices#[indices]
    print("Point cloud loaded..")
    
    return vertices



def load_mesh(mesh_file_name):
    return mesh_file_name

def setup_seeds(seed=42):
    seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    setup_seeds()

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model", default="vicuna7b_v2")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


    pc_pocessor = ULIPPCProcessor()
    image_pocessor = ClipImageEvalProcessor()
    audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=2, is_eval=False, frame_length=512)
    video_processor = AlproVideoEvalProcessor(n_frms=4, image_size=224)

    cfg_path  = {
         "vicuna13b": './configs/vicuna13b.yaml',
          "vicuna7b": './configs/vicuna7b.yaml',
         "no_init": './configs/vicuna7b_no_init.yaml',
          "projection": './configs/vicuna7b_projection.yaml',
        "vicuna7b_v2": './configs/vicuna7b_v2.yaml'
        }
    config = OmegaConf.load(cfg_path[args.model])
    print(cfg_path[args.model])
    print('Loading model...')
    model_cls = registry.get_model_class(config.get("model", None).arch)
    model =  model_cls.from_config(config.get("model", None))
    # if args.model == 'flant5xxl':
    #     model = Blip2T5MMInstruct.from_config(config.get("model", None))
    # else:
    #      model = Blip2VicunaMMInstruct.from_config(config.get("model", None))
    model.to(device)
    print('Loading model done!')


    def inference(image, point_cloud, audio, video, prompt, qformer_prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method):
        if qformer_prompt == "" or qformer_prompt == None:
            qformer_prompt = prompt
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(image, point_cloud, audio, video, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        if image is not None:
            image = image_pocessor(image).unsqueeze(0).to(device)
        if point_cloud is not None:
            point_cloud = convert_mesh_to_numpy(point_cloud)
            point_cloud = pc_pocessor(point_cloud).unsqueeze(0).to(device)
        if audio is not None:
            audio = audio_processor(audio).unsqueeze(0).to(device)
        if video is not None:
            video = video_processor(video).unsqueeze(0).to(device)
        
        samples = {"prompt": prompt}
        if image is not None:
            samples["image"] = image
        if point_cloud is not None:
            samples["pc"] = point_cloud
        if audio is not None:
            samples["audio"] = audio
        if video is not None:
            samples["video"] = video

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
            special_qformer_input_prompt=qformer_prompt
        )

        return output[0]
    
    # examples = [
    #     [None, None, './examples/audio/110714_wren.wav', None, "What do you hear?", 1, 50, 3, 1., 2.2, 0.9, "Beam Search"],
        
    #     [None, './examples/point_cloud/banana.glb', None, None, "Describe the 3D model.", 1, 250, 5, -1, 1.4, 0.9, "Beam Search"],
    #     [None, None, './examples/audio/Group_of_Dogs_Barking.wav', None, "What do you hear?", 6, 250, 5, -1., 1, 0.9, "Beam Search"],

    # ]



    # iface.launch(share=True)
    # Create a Blocks interface
    with gr.Blocks() as demo:
        # Create a row for input components
        with gr.Row():

            image_input = gr.Image(type="pil")

            pc_input = gr.Model3D()

            audio_input = gr.Audio(sources=["upload"], type="filepath")

            video_input = gr.Video()
        with gr.Row():
                prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)
                qformer_prompt_textbox = gr.Textbox(label="Qformer Prompt:", placeholder="prompt", lines=2)
        
        # Output text below the inputs
        output_text = gr.Textbox(label="Output")

        # When the button is clicked, the inference function will run
        submit_button = gr.Button("Submit")


        with gr.Row():
            min_len = gr.Slider(
                minimum=1,
                maximum=50,
                value=1,
                step=1,
                interactive=True,
                label="Min Length",
            )

            max_len = gr.Slider(
                minimum=10,
                maximum=500,
                value=250,
                step=5,
                interactive=True,
                label="Max Length",
            )
        with gr.Row():
            sampling = gr.Radio(
                choices=["Beam search", "Nucleus sampling"],
                value="Beam search",
                label="Text Decoding Method",
                interactive=True,
            )

            top_p = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="Top p",
            )

            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="Beam Size",
            )
        with gr.Row():
            len_penalty = gr.Slider(
                minimum=-1,
                maximum=2,
                value=1,
                step=0.2,
                interactive=True,
                label="Length Penalty",
            )

            repetition_penalty = gr.Slider(
                minimum=0.,
                maximum=3,
                value=1.5,
                step=0.2,
                interactive=True,
                label="Repetition Penalty",
            )
        submit_button.click(
            fn=inference,
            inputs=[image_input, pc_input, audio_input, video_input, prompt_textbox, qformer_prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling],
            outputs=output_text
        )

        # gr.Examples(examples, inputs=[image_input, pc_input, audio_input, video_input, prompt_textbox, qformer_prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling], outputs=output_text)


    demo.launch(share=True)