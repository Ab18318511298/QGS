import math
import torch
import numpy as np
import torch.nn.functional as F

# Copied from 2DGS
# 将深度图（2D）转换为相机坐标系下的 3D 点云，并计算对应像素的射线方向
def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    # c2w = torch.eye(c2w.shape[0],device = c2w.device)
    cam_intr = view.cam_intr
    W, H = view.image_width, view.image_height
    # fx = W / (2 * math.tan(view.FoVx / 2.))
    # fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[cam_intr[0], 0., cam_intr[2]],
        [0., cam_intr[1], cam_intr[3]],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float()+0.5, torch.arange(H, device='cuda').float()+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    # rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    # rays_o = c2w[:3, 3]
    
    
    rays_o = torch.zeros_like(c2w[:3, 3]) # 初始化相机坐标系原点
    rays_d = points @ intrins.inverse().T # 像素坐标通过内参的逆转置矩阵，转换为相机坐标系下的射线方向rays_d

    # 结合方向与深度，计算得到相机坐标系下的3D点坐标。
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points, rays_d

# 从深度图计算表面法向量图，每个像素对应一个归一化的法向量。并返回对应的 3D 点云
def depth_to_normal(view, depth):
    points, rays_d = depths_to_points(view, depth)
    # depth.shape[1:]排除第一维batch，剩下[H, W]两维度。“*”将[H, W]解包为H与W，最终重塑后得到形状为[H, W, 3]的points。
    points = points.reshape(*depth.shape[1:], 3) 
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def FindNeiborPoints(points):
    points = points.view(-1,3) # points形状为[N, 3]。
    from simple_knn._C import knnWithIndices
    # knnWithIndices 函数计算每个点的最近邻索引
    neibors = knnWithIndices(points) # [N, K_MAX]
    # neibor_points：每个点的邻域点集合（形状为 [N, K_MAX, 3]，K_MAX 为最大邻域点数）
    neibor_points = points[[neibors.long()]]
    return neibor_points

# Given the position of a point and its normal in the camera coordinate system, 
# construct the camera-to-points transformation matrix.
# 构建 “相机坐标系到点局部坐标系” 的变换矩阵（C2P）。局部坐标系以目标点为原点，法向量为 z 轴，构建正交的 x、y 轴。
def getC2P(points, normals):
    normals = normals.view(-1,3,1)
    tangent = torch.zeros_like(normals)
    first_mask = torch.abs(normals[:, 0, 0]) < torch.abs(normals[:, 1, 0])
    tangent[first_mask, 0, 0] = 1.0
    tangent[~first_mask, 1, 0] = 1.0
    
    local_x = torch.cross(tangent, normals)
    local_x_normalize = local_x / torch.norm(local_x, dim=1).view(-1,1,1)
    
    local_y = torch.cross(normals, local_x_normalize)
    local_y_normalize = local_y / torch.norm(local_y, dim=1).view(-1,1,1)
    rotation = torch.cat([local_x_normalize, local_y_normalize, normals], dim=-1) # [N, 3]
    translation = points.view(-1,3,1)
    
    C2P = torch.eye(4,device=points.device).unsqueeze(0).repeat(normals.shape[0], 1, 1)
    C2P[:,:3,:3] = rotation.permute(0,2,1)
    C2P[:,:3,3:4] = -rotation.permute(0,2,1) @ translation
    C2P = torch.nan_to_num(C2P,0,0)
    return C2P

def AnalysisCurvature(C2P, neibor_points):
    """
        C2P: [N, 4, 4]
        neibor_points: [N, K_MAX, 3]
        fit N quadratic surface
    """
    local_neibor_points = (C2P.unsqueeze(1)[..., :3, :3] @ neibor_points[..., None] + C2P.unsqueeze(1)[..., :3, 3:4]).squeeze() # [N, K_MAX, 3, 1]
    # local_neibor_points = neibor_points
    X = local_neibor_points[..., 0] # [N, K_MAX]
    Y = local_neibor_points[..., 1]
    Z = local_neibor_points[..., 2]
    
    A = torch.stack([torch.ones_like(X),X,Y,X**2, X*Y, Y**2], dim=2)# [N, 3, 3]
    coeffs = torch.linalg.lstsq(A, Z.unsqueeze(2)).solution # [N, 3] means a,b,c for z = ax^2 + bxy + cy^2
    hx = coeffs[:,1:2]
    hy = coeffs[:,2:3]
    hxx = 2 * coeffs[:,3:4]
    hxy = coeffs[:,4:5]
    hyy = 2 * coeffs[:,5:6]
    
    curvature = (hxx*hyy-hxy**2) / (1+hx**2+hy**2)**2
    return curvature

def points_to_curvature(points, normals, neibor_points, curv_mask):
    curv_mask = curv_mask.view(-1)
    neibor_points = FindNeiborPoints(points)
    points_with_mask = points[curv_mask]
    neibor_points_with_mask = neibor_points[curv_mask]
    normals_points_with_mask = normals[curv_mask]
    C2P = getC2P(points_with_mask, normals_points_with_mask)
    curvature = AnalysisCurvature(C2P, neibor_points_with_mask)
    return curvature
    
    
    
