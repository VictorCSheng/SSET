# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:49:01 2020

@author: CS
"""

import cv2
import numpy as np
from skimage import transform as trans


def Match_based_flann_knnMatch(des_fix, des_mov, ratio = 0.8):
    ######
    ###
    #
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #
    search_params = dict(checks = 50) #
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_fix, des_mov, k=2)  #

    ##
    good_matches = []
    good_matches_without_list = []
    for i,(m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:   #
            good_matches.append([m])
            good_matches_without_list.append(m)

    return good_matches, good_matches_without_list

def Match_based_flann_knnSearch(des_fix, des_mov, ratio = 1.5):
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(des_mov, flann_params)
    idx, dist = flann.knnSearch(des_fix, 2, params={})
    del flann
    matches = np.c_[np.arange(len(idx)), idx[:, 0]]
    pass_filter = dist[:, 0] * ratio < dist[:, 1]
    matches = matches[pass_filter]
    distance = dist[pass_filter, 0]
    matches_lst = [[cv2.DMatch(m[0], m[1], d)] for m, d in zip(matches, distance)]
    return matches_lst

def BFMatch_knnMatch(des_fix, des_mov, ratio = 0.8):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_fix, des_mov, k=2)

    good = []
    good_matches_without_list = []
    for m, n in matches:
       if m.distance < ratio * n.distance:
           good.append([m])
           good_matches_without_list.append(m)

    return good, good_matches_without_list

class Align:
    def __init__(self, moving, fixed, kp_m, kp_f, des_m, des_f):
        self.moving = moving
        self.fixed = fixed
        self.kp_m = kp_m
        self.kp_f = kp_f
        self.des_m = des_m
        self.des_f = des_f

    def get_match(self):
        raise NotImplementedError

    def get_deformation(self):
        raise NotImplementedError

    def deformation(self, relation, img, scale):
        raise NotImplementedError

    def align(self, img=None, scale=1):
        return self.deformation(self.get_deformation(), scale)

class AlignViaPro(Align):
    def __init__(self, moving, fixed, kp_m, kp_f, des_m, des_f, method="homography", ransacReprojThreshold = 3):
        super(AlignViaPro, self).__init__(moving, fixed, kp_m, kp_f, des_m, des_f)
        self.method = method
        self.ransacReprojThreshold = ransacReprojThreshold # 抛弃一个点对的阈值

        self.pts_f = None
        self.pts_m = None

    def get_match(self):
        # matches, matcheswithoutlist = Match_based_flann_knnMatch(self.des_f, self.des_m)
        matches, matcheswithoutlist = BFMatch_knnMatch(self.des_f, self.des_m)
        
#        matches = sorted(matches, key=lambda x: x[0].distance)
        
        print("matches:%d" % len(matches))
        if len(matches) <= 8:
            print('Warning: Too few good registration points!')

        pts_f = np.float32([self.kp_f[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_m = np.float32([self.kp_m[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        self.pts_f = pts_f
        self.pts_m = pts_m

        return matches, matcheswithoutlist, pts_f, pts_m

    def get_deformation(self):
        if self.pts_f.all() != None and self.pts_m.all() != None:
            pts_f = self.pts_f
            pts_m = self.pts_m
        else:
            matches, matcheswithoutlist, pts_f, pts_m = self.get_match()
        if self.method == "homography":   #
            mat, inters = cv2.findHomography(pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold = self.ransacReprojThreshold)
        elif self.method == "rigid":
            mat, inters = cv2.estimateAffinePartial2D(pts_m, pts_f, method=cv2.RANSAC,
                                                      ransacReprojThreshold=self.ransacReprojThreshold)
        else:
            mat, inters = cv2.estimateAffine2D(pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold = self.ransacReprojThreshold)
        
        rrt = self.ransacReprojThreshold
        while inters.sum() <= 15:
            rrt = rrt + 1
            
            if rrt > 100:
                print("Warning: Too few good registration points")
                break
            
            if self.method == "homography":
                mat, inters = cv2.findHomography(pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold = rrt)
            elif self.method == "rigid":
                mat, inters = cv2.estimateAffinePartial2D(pts_m, pts_f, method=cv2.RANSAC,
                                                          ransacReprojThreshold=self.ransacReprojThreshold)
            else:
                mat, inters = cv2.estimateAffine2D(pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold = rrt)
        
        print(inters.sum())
        print(mat)
        print("threshold:%d" % rrt)

        return mat, inters

    def deformation(self, mat, inters, scale = 1):
        size = tuple(self.fixed.shape)[:2][::-1]
        if self.method == "homography":
            return cv2.warpPerspective(self.moving, mat, size)
        else:
            size = (int(size[0]/scale),int(size[1]//scale))
            mat[0, 2] = mat[0, 2] / scale
            mat[1, 2] = mat[1, 2] / scale
            mat[0, 0] = 1
            mat[0, 1] = 0
            mat[1, 0] = 0
            mat[1, 1] = 1
            return cv2.warpAffine(self.moving, mat, size)