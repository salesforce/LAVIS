import gradio as gr
from lavis.models import load_model_and_preprocess
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    args = parser.parse_args()

    image_input = gr.Image(type="pil")

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

    len_penalty = gr.Slider(
        minimum=-1,
        maximum=2,
        value=1,
        step=0.2,
        interactive=True,
        label="Length Penalty",
    )

    repetition_penalty = gr.Slider(
        minimum=-1,
        maximum=3,
        value=1,
        step=0.2,
        interactive=True,
        label="Repetition Penalty",
    )


    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Loading model...')

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )

    print('Loading model done!')

    def inference(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method, modeltype):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )

        return output[0]

    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling],
        outputs="text",
        allow_flagging="never",
    ).launch()
