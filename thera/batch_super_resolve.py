#!/usr/bin/env python

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
import logging

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from model import build_thera
from utils import make_grid, interpolate_grid
from super_resolve import process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_folder(input_folder, output_folder, checkpoint, scale=None, size=None, no_ensemble=False, patch=None):
    """Process all images in a folder for super-resolution."""
    logging.info(f"Starting processing for folder: {input_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Output folder: {output_folder}")

    # Load the model checkpoint
    try:
        with open(checkpoint, 'rb') as fh:
            check = pickle.load(fh)
            params, backbone, size_model = check['model'], check['backbone'], check['size']
        logging.info("Model checkpoint loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return

    model = build_thera(3, backbone, size_model)

    # Process each image in the input folder
    num_files = len(os.listdir(input_folder))
    logging.info(f"Found {num_files} files in the input folder.")
    if num_files == 0:
        logging.warning("No files found in the input folder.")
        return
    
    for file_number, file_name in enumerate(os.listdir(input_folder), start=1):
        logging.info(f"Processing file {file_number}/{num_files}: {file_name}")
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Skip non-image files
        if not file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif')):
            logging.warning(f"Skipping non-image file: {file_name}")
            continue

        # logging.info(f"Processing {input_path}...")

        try:
            # Read and process the image
            source = np.asarray(Image.open(input_path).convert('RGB')) / 255.

            if scale is not None:
                if size is not None:
                    raise ValueError('Cannot specify both size and scale')
                target_shape = (
                    round(source.shape[0] * scale),
                    round(source.shape[1] * scale),
                )
            elif size is not None:
                target_shape = size
            else:
                raise ValueError('Must specify either size or scale')

            out = process(source, model, params, target_shape, not no_ensemble, patch)

            # Save the output image
            Image.fromarray(np.asarray(out)).save(output_path)
            # logging.info(f"Saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to process {input_path}: {e}")

    logging.info("Processing completed.")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('input_folder', help='Path to the input folder containing images')
    parser.add_argument('output_folder', help='Path to the output folder to save super-resolved images')
    parser.add_argument('--scale', type=float, help='Scale factor for super-resolution')
    parser.add_argument('--size', type=int, nargs=2,
                        help='Target size (h, w), mutually exclusive with --scale')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable geo-ensemble')
    parser.add_argument('--patch', type=int, default=None, help='Patch size of input image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_folder(args.input_folder, args.output_folder, args.checkpoint, args.scale, args.size, args.no_ensemble, args.patch)
