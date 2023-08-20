"""
TODO: Add description, gradio demo
"""
from argparse import ArgumentParser, Namespace

import gradio as gr
from PIL import Image
from easyfaret import EasyFaceRetreival


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="data")
    parser.add_argument("-q", "--queries", nargs="+", type=str, default=[])
    parser.add_argument(
        "--share", action="store_true", help="share your model on gradio"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="host address to serve model on"
    )
    parser.add_argument("--port", type=int, default=7860, help="port to serve model on")
    parser.add_argument("--debug", action="store_true", help="debug mode for testing")
    return parser.parse_args()


def main():
    args = get_args()
    if len(args.queries) > 0:
        retreival = EasyFaceRetreival.from_queries(args.queries)
    else:
        retreival = EasyFaceRetreival.from_images(args.path)

    def search(image):
        results = retreival.search(image, n_results=1)
        if len(results) == 0:
            return None, "얼굴을 찾을 수 없습니다"

        image = Image.open(results[0]["file"])
        tags = ""
        for k, v in results[0].items():
            tags += f"{k}: {v}\n"
        return image, tags

    iface = gr.Interface(
        fn=search,
        inputs=gr.Image(type="pil", label="Input image"),
        outputs=[gr.Image(label="Results"), gr.Textbox(label="tags")],
    )

    iface.launch(
        debug=args.debug,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
