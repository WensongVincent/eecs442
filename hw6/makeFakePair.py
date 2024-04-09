#!/usr/bin/python

import numpy as np
import cv2
import os
import pdb


def getImages(src):
    images = []
    for fn in os.listdir(src):
        images.append(cv2.imread(os.path.join(src,fn)))
    return images

def getScene(images):
    #take images and make (nominalz,ul,ur,br,bl,image)
    np.random.seed(42)
    billboards = []
    for im in images:
        x = np.random.random()*6-3
        y = np.random.random()*6-3
        z = np.random.random()*5+5
        scale = np.random.random()*0.1+1
        H,W = im.shape[0], im.shape[1]
        
        HScale, WScale = H/max(H,W)*scale, W/max(H,W)*scale

        ul, ur = np.array([[x,y,z]]).T, np.array([[x+WScale,y,z]]).T
        br, bl = np.array([[x+WScale,y+HScale,z]]).T, np.array([[x,y+HScale,z]]).T
        billboards.append((z,ul,ur,br,bl,im))
    return billboards 


def getStarmap(H,W):
    """Generate a starmap"""
    scolor = [10,10,10]
    I = np.dstack([np.ones((H,W),dtype=np.uint8)*scolor[2-i] for i in range(len(scolor))])

    for i in range(1000):
        i, j = int(np.random.random()*H), int(np.random.random()*W)
        c = 255-int(np.random.random()*256)
        I = cv2.circle(I, (j,i), int(np.random.random()*1+1), (c,c,c),-1)
    return I

def renderBillboards(H,W,M,billboards, starter):
    """Render billboards and return corners for tracking"""
    order = list(range(len(billboards)))
    order.sort(key=lambda i:-billboards[i][0])

    def prv(v):
        res = M[:,:3]@v+M[:,3:]
        return res[:2] / res[2]

    def isBehind(v):
        res = M[:,:3]@v+M[:,3:]
        return res[2] < 0

    I = starter.copy()

    allCorners, all3D = [], []

    for ii, i in enumerate(order):
        znom, ul, ur, br, bl, im = billboards[i]
        imh, imw = im.shape[0], im.shape[1]
        ulp, urp, brp, blp = prv(ul), prv(ur), prv(br), prv(bl)
        
        if isBehind(ul) or isBehind(ur) or isBehind(br) or isBehind(bl):
            continue

        uls, urs, brs, bls = np.array([[0.0,0]]).T, np.array([[imw,0.0]]).T, np.array([[imw,imh*1.0]]).T, np.array([[0.0,imh]]).T

        pts1 = np.hstack([ulp,urp,brp,blp]).T
        pts2 = np.hstack([uls,urs,brs,bls]).T

        allCorners.append(pts1)
        all3D.append(np.hstack([ul,ur,br,bl]).T)
        for i in range(10):
            frac = (i+1) / 11.0
            top, right = ul*frac+ur*(1-frac), ur*frac+br*(1-frac)
            bottom, left = br*frac+bl*(1-frac), bl*frac+ul*(1-frac)
            allCorners.append(np.hstack([prv(top),prv(right),prv(bottom),prv(left)]).T)
            all3D.append(np.hstack([top,right,bottom,left]).T)

        warp = cv2.getPerspectiveTransform( pts2.astype(np.float32), pts1.astype(np.float32))
        WI = cv2.warpPerspective(im,warp,(W,H))
        WM = (cv2.warpPerspective(np.ones((imh,imw),dtype=np.uint8)*255,warp,(W,H))>128).astype(np.float)
        for c in range(3):
            I[:,:,c] = ((WM*WI[:,:,c])+((1-WM)*I[:,:,c])).astype(np.uint8)
    return I, np.vstack(allCorners), np.vstack(all3D)


def dumpRender(targetName,R1,R2,t1,t2):

    if not os.path.exists(targetName):
        os.mkdir(targetName)

    H, W = 960, 1280
    K = np.diag([1000,1000,1]); K[0,2] = W/2; K[1,2] = H/2


    billboards = getScene(getImages("gallery/"))

    start = getStarmap(H,W)
    b1, c1, c3d1 = renderBillboards(H,W,K@np.hstack([R1,t1]),billboards,start)
    b2, c2, c3d2 = renderBillboards(H,W,K@np.hstack([R2,t2]),billboards,start)
        
    cv2.imwrite(os.path.join(targetName,"im1.png"),b1)
    cv2.imwrite(os.path.join(targetName,"im2.png"),b2)
    np.savez(os.path.join(targetName,"data.npz"),pts1=c1.astype(np.int),pts2=c2.astype(np.int),pts1_3D=c3d1,pts2_3d=c3d2,K1=K,K2=K)


def dumpScaledMotion(targetName,R,t,n,tstart=np.zeros((3,1))):
    
    if not os.path.exists(targetName):
        os.mkdir(targetName)

    H, W = 960, 1280
    start = getStarmap(H,W)
    billboards = getScene(getImages("gallery/"))
    for i in range(n):
        K = np.diag([1000,1000,1]); K[0,2] = W/2; K[1,2] = H/2
        b1, _, _ = renderBillboards(H,W,K@np.hstack([R,t*i+tstart]),billboards,start)
        cv2.imwrite(os.path.join(targetName,"frame_%06d.png" % i), b1)
    
    os.system("ffmpeg -i %s/frame_%%06d.png -qscale 0 %s/vid.wmv" % (targetName,targetName))

def xzrot(theta):
    R = np.eye(3)
    R[0,0] = np.cos(theta); R[0,2] = -np.sin(theta)
    R[2,0] = np.sin(theta); R[2,2] = np.cos(theta)
    return R


if __name__ == "__main__":
    RR = np.eye(3); RL = np.eye(3); RRLess = np.eye(3)
    theta = 0.349066
    RR[0,0] = np.cos(theta); RR[0,2] = -np.sin(theta)
    RR[2,0] = np.sin(theta); RR[2,2] = np.cos(theta)

    RR = xzrot(theta)
    RRLess = xzrot(theta/2)
    RL = xzrot(-theta)

    RRMore = xzrot(theta*3.25)
    RLMore = xzrot(-theta*3.25)

    if 1:
        t1 = np.zeros((3,1)); t2 = np.zeros((3,1))
        t1[0] = 4; t1[2] = 4; t2[0] = -4; t2[2] = 4
        dumpRender("task23/reallyInwards/", RRMore, RLMore, t1, t2)


    if 0:
        t1 = np.zeros((3,1)); t2 = np.zeros((3,1)); t2[2] = 1; t2[0] = -0.5; t2[1] = -0.5; t2[1] = -1
        dumpRender("task23/xyztrans/",np.eye(3),np.eye(3),t1,t2)

        t1 = np.zeros((3,1)); t2 = np.zeros((3,1)); t2[0] = -0.5
        dumpRender("task23/xtrans/",np.eye(3),np.eye(3),t1,t2)

        t1 = np.zeros((3,1)); t2 = np.zeros((3,1)); t2[1] = -0.5
        dumpRender("task23/ytrans/",np.eye(3),np.eye(3),t1,t2)

        t1 = np.zeros((3,1)); t2 = np.zeros((3,1)); t2[2] = -1.5
        dumpRender("task23/ztrans/",np.eye(3),np.eye(3),t1,t2)

        t1 = np.zeros((3,1)); t1[0] = 6; t1[2] = 2; t2 = np.zeros((3,1)); t2[0] = -6; t2[2] = 2
        dumpRender("task23/inwards/", RR, RL, t1, t2)

        tx = np.zeros((3,1)); tx[0] = -0.1
        tz = np.zeros((3,1)); tz[2] = -0.1

        ts = np.zeros((3,1)); ts[0] = 2; ts[2] = 2
        dumpRender("task23/zrtrans/",RR,RR,ts,ts+20*tz)

        ts = np.zeros((3,1)); ts[0] = 2; ts[2] = 2
        dumpRender("task23/zrtransrot/",RR,RRLess,ts,ts+20*tz)


    if 0:
        dumpScaledMotion("temp/ztrans/",np.eye(3),tz,80)

        dumpScaledMotion("temp/xtrans/",np.eye(3),tx,80)
    
        dumpScaledMotion("temp/zrtrans/",RR,tz,80,tstart=ts)





