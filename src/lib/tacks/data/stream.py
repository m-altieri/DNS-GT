# -*- coding: utf-8 -*-
"""Classes and functions for handling streams.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
from threading import Thread
from time import sleep

import cv2
import numpy as np


class WebcamStream:
    """Return the stream of a webcam.

    Parameters
    ----------
    crop_area : list of slices or None, optional
        Slices to apply on each spatial dimension (default: None).
    logger : logging.Logger, optional
        Logging system (default: None).
    """

    def __init__(self, crop_area=None, logger=None):

        self.logger = logger

        if crop_area is None:
            crop_area = [slice(None), slice(None)]
        self.crop_area = crop_area

        # init the stream
        self.cap = cv2.VideoCapture(0)

        # test presence of a webcam
        if not self.cap.isOpened():
            err_msg = 'Could not open webcam stream.'
            raise ValueError(err_msg)

        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # get the width and height of the frame, including cropping area
        bounds_height = crop_area[0].indices(frame_height)
        bounds_width = crop_area[1].indices(frame_width)

        self.frame_size = (
            bounds_height[1] - bounds_height[0],
            bounds_width[1] - bounds_width[0],
        )

        _ = self.cap.read()

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.frame = None
        self.is_working = False

    def get_frame(self, n_avg_frames=5):
        """Return the current frame resized to the desired size.

        Parameters
        ----------
        n_avg_frames : int
            Number of frames for averaging.

        Returns
        -------
        array-like
            Frame as an array of type uint8 in WHC format.
        """

        frame = self.frame.astype(np.float32) / 255
        for _ in range(n_avg_frames - 1):
            frame += self.frame.astype(np.float32) / 255
            sleep(0.001)

        frame = frame / n_avg_frames

        # convert frame into RGB colors
        frame = (frame * 256).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # crop the frame
        if self.crop_area is not None:
            frame = frame[self.crop_area[0], self.crop_area[1]]

        return frame

    def update(self):
        """Update the current frame."""
        while self.cap.isOpened():
            (self.is_working, self.frame) = self.cap.read()

    def stop(self):
        """Stop the webcam."""
        self.cap.release()
