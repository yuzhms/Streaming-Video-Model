import sys

import click
import imageio
from path import Path
from tqdm.contrib.concurrent import process_map
import cv2
from PIL import Image


def extract(args):
    video, frames_dir, first_frame = args
    out_seq_path = frames_dir / video.basename().split('.')[0] / 'img1'
    if not out_seq_path.exists():
        out_seq_path.makedirs()
    elif len(out_seq_path.files()) == 1800:
        print(f'▸ skip \'{Path(video).abspath()}\'')
        return

    reader = imageio.get_reader(video)
    print(f'▸ extracting frames of \'{Path(video).abspath()}\'')
    for frame_number, image in enumerate(reader):
        n = first_frame + frame_number
        imageio.imwrite(out_seq_path / f'{n:06}.jpg', image)

H1 = 'directory where you want to save the extracted frames'
H2 = 'number from which to start counting the video frames; DEFAULT = 1'
H3 = 'the format to use to save the images/frames; DEFAULT = jpg'

# check python version
assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'

@click.command()
@click.option('--task', '-t', type=click.Choice(['extract',]), default='extract', help='task to perform')
@click.option('--input_dir_path', type=click.Path(), )
@click.option('--out_dir_path', type=click.Path(), prompt='Enter \'out_dir_path\'', help=H1)
@click.option('--first_frame', type=int, default=1, help=H2)
@click.option('--img_format', type=str, default='jpg', help=H3)
def main(task, input_dir_path, out_dir_path, first_frame, img_format):
    # type: (str, str, str, int, str) -> None
    """
    Script that splits all the videos into frames and saves them
    in a specified directory with the desired format
    """
    out_dir_path = Path(out_dir_path)
    if not out_dir_path.exists():
        out_dir_path.makedirs()
    dir = Path(input_dir_path)
    videos = dir.files()

    if task == 'extract':
        process_map(extract, zip(videos, [out_dir_path] * len(videos), [first_frame] * len(videos)))

if __name__ == '__main__':
    main()