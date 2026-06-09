from tqdm.auto import tqdm
import timeit
import requests
import os
import cv2
import numpy as np
import torch
import ffmpeg
from bsrgan_utils import uint2tensor4, tensor2uint


def time_execution(stmt, globals=None, number=1, repeat=5):
    """times the execution of a statement

    Args:
        stmt (callable): a callable object that will be timed
        globals (Any, optional): Global variables. Defaults to None.
        number (int, optional): Number of times to run the statement. Defaults to 1.
        repeat (int, optional): Number of times to repeat the timing. Defaults to 5.

    Returns:
        tuple: Execution times (List[float]), Mean (float), Standard deviation (float)
    """
    try:
        times = timeit.repeat(stmt, globals=globals, number=number, repeat=repeat)

        mean = np.mean(times)
        std = np.std(times)

        print(f"{mean:.2f} ± {std:.2f} seconds")

        return times, mean, std
    except Exception as e:
        print("Error timing execution")
        raise e


def preprocess(frame: np.ndarray) -> torch.Tensor:
    """preprocesses a frame to convert it to a format that can be fed to the model

    Args:
        frame (np.ndarray): NumPy array of the frame

    Returns:
        torch.Tensor: PyTorch tensor of the frame
    """
    try:
        return uint2tensor4(frame)
    except Exception as e:
        print("Error converting frame to tensor")
        raise e


def postprocess(frame: torch.Tensor) -> np.ndarray:
    """postprocesses a frame to convert it to a format that can be displayed

    Args:
        frame (torch.Tensor): PyTorch tensor of the frame

    Returns:
        np.ndarray: NumPy array of the frame
    """
    try:
        res = tensor2uint(frame)
        return res
    except Exception as e:
        print("Error converting tensor to frame")
        raise e


def collect_all_frames(video: cv2.VideoCapture) -> list[np.ndarray]:
    """collects all the frames from a video

    Args:
        video (cv2.VideoCapture): Video capture object

    Returns:
        list[np.ndarray]: List of frames
    """
    try:
        frames = []

        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=num_frames, desc="Collecting frames from video")

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frames.append(frame)
            pbar.update(1)

        return frames
    except Exception as e:
        print("Error collecting frames")
        raise e


def write_all_frames(frames: list[np.ndarray], output_video: cv2.VideoWriter) -> None:
    """writes all the frames to an output video

    Args:
        frames (list[np.ndarray]): List of frames
        output_video (cv2.VideoWriter): Video writer object
    """
    try:
        for frame in tqdm(frames, desc="Writing frames to output video"):
            output_video.write(frame)
    except Exception as e:
        print("Error writing frames to output video")
        raise e


def download_file(url, output_file):
    """downloads a file from the given URL and saves it to the specified output file

    Args:
        url: The URL of the file to download
        output_file: The path where the downloaded file will be saved

    Returns:
        bool: True if the download was successful, False otherwise
    """

    def _do_download(verify_ssl):
        response = requests.get(url, stream=True, timeout=10, verify=verify_ssl)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    try:
        _do_download(verify_ssl=True)
        print(f"File has been downloaded as {output_file}")
        return True
    except requests.exceptions.SSLError:
        print("SSL verification failed, retrying without SSL verification...")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            _do_download(verify_ssl=False)
            print(f"File has been downloaded as {output_file}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def download_file_to_memory(url):
    """Downloads a file from the given URL and returns its raw content as bytes.

    Args:
        url: The URL of the file to download

    Returns:
        bytes on success, or None on failure
    """

    def _fetch(verify_ssl):
        response = requests.get(url, stream=True, timeout=30, verify=verify_ssl)
        response.raise_for_status()
        return response.content

    try:
        return _fetch(verify_ssl=True)
    except requests.exceptions.SSLError:
        print("SSL verification failed, retrying without SSL verification...")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return _fetch(verify_ssl=False)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None


def resize_video(input_path, scale=2):
    """reduces the resolution of the given video by a given scale factor, \
        so the output video will have a resolution that is 1 / scale times \
        the original resolution

    Args:
        input_path: The file path to the input video
        scale: The factor by which to scale the video resolution. \
            The output size will be original_size / scale. Default value is 2
    """

    try:
        original_video_info = ffmpeg.probe(input_path)
        original_video_height = original_video_info["streams"][0]["height"]
        original_video_width = original_video_info["streams"][0]["width"]

        new_height = int(original_video_height / scale)
        new_width = int(original_video_width / scale)

        temp_output_path = "temp_resized_video.mp4"

        ffmpeg.input(input_path).output(temp_output_path, vf=f"scale={new_width}:{new_height}").overwrite_output().run()

        # After processing, replace the original file with the resized one
        os.replace(temp_output_path, input_path)
        print(f"Video resized successfully and overwritten: {input_path}")

    except ffmpeg.Error as e:
        print(f"Error: {e}")
        print("ffmpeg stderr output:\n", e.stderr.decode("utf-8"))
