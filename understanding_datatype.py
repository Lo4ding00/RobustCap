import torch
from torch.utils.data import DataLoader
import articulate as art
import config
from articulate.utils.torch import *
from articulate.utils.print import *
from config import *
import tqdm
import numpy as np
import utils
import os
from net.smplify.run import smplify_runner
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file)
J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float()
run_smplify = False


def cal_mpjpe(pose, gt_pose, cal_pampjpe=False):
    _, gt_joints, gt_vertices = body_model.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, joints, vertices = body_model.forward_kinematics(pose.cpu(), calc_mesh=True)
    keypoints_3d = torch.matmul(J_regressor_batch, vertices)[:, :14]
    pred_pelvis = keypoints_3d[:, [0], :].clone()
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    keypoints_3d = keypoints_3d - pred_pelvis
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    if cal_pampjpe:
        pampjpe = utils.reconstruction_error(keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        return torch.tensor(
            [(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean(),
             pampjpe.mean()])
    return (gt_joints - joints).norm(dim=2)


if __name__ == '__main__':
    sequence_idx = 13
    i = sequence_idx

    dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))

    name_sequence = dataset['name'][i]
    joint3d = dataset['joint3d'][i]  # 相机坐标下，人体各关节的三维坐标
    tranc = dataset['tranc'][i]  # 相机坐标系下，人体坐标系的平移
    cam_T = dataset['cam_T'][i]  # 齐次变换矩阵

    dataset = torch.load(os.path.join('data/dataset_work/3DPW/', 'test' + '.pt'))
    data, label = [], []
    i = sequence_idx

    print(dataset['name'][i])

    gt_result_name = os.path.join(name_sequence + '_gt_result.pt')

    Kinv = dataset['cam_K'][i].inverse()
    oric = dataset['imu_oric'][i]
    accc = dataset['imu_accc'][i]
    j2dc = torch.zeros(len(oric), 33, 3)
    j2dc[..., :2] = dataset['joint2d_mp'][i][..., :2]
    j2dc[..., -1] = dataset['joint2d_mp'][i][..., -1]
    pose = dataset['posec'][i].view(-1, 24, 3, 3)
    tran = dataset['tranc'][i].view(-1, 3)
    data.append(torch.cat((j2dc.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
    label.append(torch.cat((tran, pose.flatten(1)), dim=1))

    from net.sig_mp import Net

    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.use_flat_floor = False
    net.eval()

    test_dataloader = DataLoader(RNNDataset(data, label, split_size=-1, device=device), 32,
                                 collate_fn=RNNDataset.collate_fn)

    pose_p, tran_p, pose_t, tran_t = [], [], [], []

    print('\rRunning network')
    batch, seq = 0, 0
    for d, l in test_dataloader:
        batch_pose, batch_tran = [], []
        for i in range(len(d)):
            pose, tran = [], []
            K = dataset['cam_K'][sequence_idx].to(device)
            j2dc = d[i][:, :99].reshape(-1, 33, 3).to(device)
            j2dc = K.inverse().matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
            j2dc[..., -1] = d[i][:, :99].reshape(-1, 33, 3)[..., -1]
            first_tran = l[i][0, :3].reshape(3)

            d[i][:][99:105] = 0
            d[i][:][111:114] = 0
            d[i][:][117:135] = 0
            d[i][:][153:162] = 0
            j2dc = torch.zeros(2898, 33, 3)

            for j in tqdm.trange(len((d[i]))):
                Tcw = dataset['cam_T'][sequence_idx][j][:3, :3]
                net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
                if j == 0:
                    p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                              d[i][j][117:].reshape(6, 3, 3), first_tran)
                else:
                    p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                              d[i][j][117:].reshape(6, 3, 3))
                pose.append(p)
                tran.append(t)
            seq += 1
            pose, tran = torch.stack(pose), torch.stack(tran)
            if run_smplify:
                j2dc_opt = d[i][:, :99].reshape(-1, 33, 3)
                oric = d[i][:, 117:].reshape(-1, 6, 3, 3)
                pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001,
                                                    use_lbfgs=True, opt_steps=1, cam_k=K)
            batch_pose.append(pose)
            batch_tran.append(tran)
            net.reset_states()
        pose_p.extend(batch_pose)
        tran_p.extend(batch_tran)
        pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
        tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
        batch += 1

    errors = cal_mpjpe(pose_p[0], pose_t[0], cal_pampjpe=False)
    print('weaker_pose_result', errors.mean(dim=0))
