from inference import Inference
import argparse
#import gradio as gr
#import glob
#from huggingface_hub import hf_hub_download
#import os

import json 

def parse_option():
    parser = argparse.ArgumentParser('MetaFG Inference script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')
    # easy config modification
    parser.add_argument('--model-path', type=str, help="path to model data")
    parser.add_argument('--img-size', type=int, default=384, help='path to image')
    parser.add_argument('--meta-path', default="meta.txt", type=str, help='path to meta data')
    parser.add_argument('--names-path', type=str, help='path to meta data')
    parser.add_argument('--image-path', type=str, help='path to image file')
    parser.add_argument('--output-path', type=str, help='path to output file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_option()

    if not args.model_path:
        model_path = "raw.inat_sgd_6k.pth"
#        model_path = hf_hub_download(repo_id="joshvm/inaturalist_sgd_4k",
#            filename="inat_sgd_6k.pth",
#            token=os.environ["HUGGINGFACE_TOKEN"])"""
    else:
        model_path = args.model_path

    if not args.cfg:
        model_config = "MetaFG_2_384_inat.yaml"
#       model_config = hf_hub_download(repo_id="joshvm/inaturalist_sgd_4k",
#                                      filename="MetaFG_2_384_inat.yaml",
#                                       token=os.environ["HUGGINGFACE_TOKEN"])
    else:
        model_config = args.cfg

    if not args.names_path:
        names_path = "inat_sgd_names.txt"
#        names_path = hf_hub_download(repo_id="joshvm/inaturalist_sgd_4k",
#                                     filename="inat_sgd_names.txt",
#                                     token=os.environ["HUGGINGFACE_TOKEN"])
    else:
        names_path = args.names_path

    if not args.image_path:
        image_path = "example_images/Malayan-colugo.jpg"
    else:
        image_path = args.image_path

    if not args.output_path:
        output_path = "/outputs/preds.json"
    else:
        output_path = args.output_path


    model = Inference(config_path=model_config,
                       model_path=model_path,
                       names_path=names_path)
    
    def classify(image):
        preds = model.infer(img_path=image, meta_data_path="meta.txt")
        #confidences = {c: float(preds[i]) for i,c in enumerate(model.classes)}

        return preds
    
    dict = {
        "input_image": image_path,
        "predictions": classify(image_path)
    }
    with open(output_path, "w") as outfile:
        json_object = json.dumps(dict, indent = 4)
        outfile.write(json_object)
#        json.dump(classify(args.image_path), outfile)
    outfile.close()
    
#    gr.Interface(fn=classify, 
#            inputs=gr.Image(shape=(args.img_size, args.img_size), type="pil"),
#            outputs=gr.Label(num_top_classes=10),
#            examples=glob.glob("./example_images/*.jpg")).launch()
