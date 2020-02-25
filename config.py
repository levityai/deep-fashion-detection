"""
Script to construct TensorFlow configuration protobuf files based on templates.
"""

import argparse
from string import Template
import utils


def main(settings):

    with open(settings.template_path, 'r') as file:
        raw_template = file.read()

    template = Template(raw_template)

    num_eval_examples = utils.get_num_eval_examples()

    config = template.substitute(
        {
            "train_num_steps": settings.train_num_steps,
            "eval_config_num_examples": num_eval_examples,
            "eval_config_num_visualizations": num_eval_examples,
            "num_classes": utils.get_num_classes()
        }
    )

    with open(settings.output_path, 'w') as file:
        file.write(config)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--train_num_steps', type=int, default=200000)
    settings = parser.parse_args()
    main(settings)


if __name__ == "__main__":
    cli()
