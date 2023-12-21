#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: infere.py
@time: 2023/12/16 14:47
@desc:
"""
from argparse import Namespace
from glob import glob
import shutil
from time import strftime
import os, sys, time

import torch

from .utils.preprocess import CropAndExtract
from .test_audio2coeff import Audio2Coeff
from .facerender.animate import AnimateFromCoeff
from .generate_batch import get_data
from .generate_facerender_batch import get_facerender_data
from .utils.init_path import init_path


def init():
    args = Namespace(
        ref_eyeblink=None,
        ref_pose=None,
        pose_style=0,
        batch_size=2,
        size=256,
        expression_scale=1,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        enhancer=None,
        background_enhancer=None,
        face3dvis=False,
        still=False,
        preprocess="crop",
        verbose=False,
        old_version=False,
        use_last_fc=False,
        bfm_folder="./checkpoints/BFM_Fitting/",
        bfm_model="BFM_model_front.mat",
        focal=1015.,
        center=112.,
        camera_d=10.,
        z_near=5.,
        z_far=15.,
    )
    return args


def out_main(audio_path) -> str:
    args = init()

    return main(pic_path, audio_path, args)


class Avatar:
    def __init__(self):
        args = init()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_root_path = os.path.dirname(__file__)
        sadtalker_paths = init_path(os.path.join(current_root_path, 'weights'),
                                    os.path.join(current_root_path, 'config'),
                                    args.size,
                                    args.old_version, args.preprocess)
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
        self.pic_path = "/mnt/d4t/workspace/red/src/sadtalk/assets/people_0.png"
        self.args = args
        self.device = device

    def run(self, audio_path: str, preprocess: str = "crop", size: int = 256) -> str:
        batch_size = self.args.batch_size
        input_yaw_list = self.args.input_yaw
        input_pitch_list = self.args.input_pitch
        input_roll_list = self.args.input_roll
        ref_eyeblink = self.args.ref_eyeblink
        ref_pose = self.args.ref_pose
        save_dir = os.path.join("/mnt/d4t/workspace/red/results", strftime("%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(self.pic_path, first_frame_dir,
                                                                                    preprocess, source_image_flag=True,
                                                                                    pic_size=size)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        ref_eyeblink = None
        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir,
                                                                           preprocess, source_image_flag=False)
        else:
            ref_eyeblink_coeff_path = None

        ref_pose = None
        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ = self.preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess,
                                                                           source_image_flag=False)
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path, still=self.args.still)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, self.args.pose_style, ref_pose_coeff_path)

        # 3dface render
        if self.args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path,
                               os.path.join(save_dir, '3dface.mp4'))

        # coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                   batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                   expression_scale=self.args.expression_scale, still_mode=self.args.still,
                                   preprocess=preprocess, size=size)

        result = self.animate_from_coeff.generate(data, save_dir, self.pic_path, crop_info,
                                                  enhancer=self.args.enhancer,
                                                  background_enhancer=self.args.background_enhancer,
                                                  preprocess=preprocess, img_size=size)

        shutil.move(result, save_dir + '.mp4')
        print('The generated video is named:', save_dir + '.mp4')

        if not self.args.verbose:
            shutil.rmtree(save_dir)

        return save_dir + ".mp4"


_avatar = None


def get_avatar() -> Avatar:
    global _avatar
    if _avatar is None:
        _avatar = Avatar()
    return _avatar
