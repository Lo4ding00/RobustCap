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
    _, _, gt_vertices = body_model.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, _, vertices = body_model.forward_kinematics(pose.cpu(), calc_mesh=True)
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
    return torch.tensor(
        [(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean()])

    # _, gt_joint_pos = body_model.forward_kinematics(gt_pose.cpu(), calc_mesh=False)
    # _, joint_pos = body_model.forward_kinematics(pose.cpu(), calc_mesh=False)
    # return (gt_joint_pos - joint_pos)


def generate_video(name_sequence):
    # 设置图像文件夹路径
    video_name = os.path.join(name_sequence + '_origin_video.mp4')
    if os.path.exists(video_name):
        return video_name

    image_folder = os.path.join('../imageFiles/imageFiles/', name_sequence)

    # 获取所有图像文件的文件名
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 设置视频文件的名称和编解码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 创建视频写入器
    out = cv2.VideoWriter(video_name, fourcc, 60, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    # 释放视频写入器
    out.release()

    print("视频已生成：" + video_name)

    return video_name


def evaluate_pw3d_ours(sequence_idx, occ=False):
    # print('Reading %s dataset "%s"' % ('test', 'data/dataset_work/3DPW/'))
    dataset = torch.load(os.path.join('data/dataset_work/3DPW/', 'test' + '.pt'))
    data, label = [], []

    i = sequence_idx

    print(dataset['name'][i])

    name_sequence = dataset['name'][i]
    if run_smplify == False:
        result_name = os.path.join(name_sequence + '_result.pt')
    else:
        result_name = os.path.join(name_sequence + '_smplify_result.pt')

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

    # 绘图必须的utility，读取视频相关
    video_name = os.path.join(name_sequence + '_origin_video.mp4')
    video = cv2.VideoCapture(video_name)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render = art.Renderer(resolution=(frame_width, frame_height), official_model_file=paths.smpl_file)
    image_directory = os.path.join('../imageFiles/imageFiles/', name_sequence)
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

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

            # IMU数据是60帧的，为了和图像数据对齐，进行切片
            j2dc = j2dc[::2, :, :]
            data = d[i][::2, :]

            label = l[i][::2, :]
            gt_trans = label[:, :3].reshape(-1, 3).to('cpu')

            for j in tqdm.trange(len((data))):
                Tcw = dataset['cam_T'][sequence_idx][j][:3, :3]
                net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
                if j == 0:
                    p, t = net.forward_online(j2dc[j].reshape(33, 3), data[j][99:117].reshape(6, 3),
                                              data[j][117:].reshape(6, 3, 3), first_tran)
                else:
                    p, t = net.forward_online(j2dc[j].reshape(33, 3), data[j][99:117].reshape(6, 3),
                                              data[j][117:].reshape(6, 3, 3))
                pose.append(p)
                tran.append(t)


                # 以下是进行绘图
                input_path = os.path.join(image_directory, image_files[j])
                im = cv2.imread(input_path)
                gt_tran = gt_trans[j]

                # # 绘制3d mesh
                # verts = body_model.forward_kinematics(p.view(-1, 24, 3, 3), tran=t.view(-1, 3), calc_mesh=True)[2][0]
                # img = render.render(im, verts, K)

                # 只绘制joints
                joints = body_model.forward_kinematics(p.view(-1, 24, 3, 3), tran=gt_tran.view(-1, 3), calc_mesh=False)[1][0]
                img = render.my_joint_render(im, joints, K)

                cv2.imshow('f', img)
                cv2.waitKey(1)

            pose, tran = torch.stack(pose), torch.stack(tran)
            batch_pose.append(pose)
            batch_tran.append(tran)
            net.reset_states()
        pose_p.extend(batch_pose)
        tran_p.extend(batch_tran)
        pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
        tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
        batch += 1
        # torch.save([pose_p, tran_p], result_name)

    # torch.save([pose_t, tran_t], gt_result_name)

    # IMU_result = tran_p[0].numpy()
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(16, 9))
    # plt.scatter(range(len(IMU_result)), IMU_result[:, 2], s=10, label='pnp', color='red')
    # plt.show()

    errors = cal_mpjpe(pose_p[0], pose_t[0], cal_pampjpe=True)
    print('mpjpe, pve:', errors.mean(dim=0))

    return name_sequence


def view(index, name_sequence, seq_idx=0, cam_idx=0, vis=True):
    dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))
    Kinv = dataset['cam_K'][index].inverse()

    if run_smplify == False:
        output_video_name = os.path.join(name_sequence + '_result.mp4')
        result_name = os.path.join(name_sequence + '_result.pt')
    else:
        output_video_name = os.path.join(name_sequence + '_smplify_result.mp4')
        result_name = os.path.join(name_sequence + '_smplify_result.pt')

    gt_result_name = os.path.join(name_sequence + '_gt_result.pt')

    if os.path.exists(output_video_name):
        return 0

    video_name = os.path.join(name_sequence + '_origin_video.mp4')

    pose_p, tran_p = torch.load(result_name)
    pose_t, tran_t = torch.load(gt_result_name)
    # you can use this command to view the result by open3d, but without overlay.
    # body_model.view_motion(pose_p, tran_p)

    video = cv2.VideoCapture(video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    render = art.Renderer(resolution=(frame_width, frame_height), official_model_file=paths.smpl_file)
    f = 0

    pbar = tqdm.tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing Frames', dynamic_ncols=True)

    while True:
        im = video.read()[1]
        if im is None:
            break
        verts = \
            body_model.forward_kinematics(pose_p[0][f].view(-1, 24, 3, 3), tran=tran_t[0][f].view(-1, 3),
                                          calc_mesh=True)[
                2][0]
        img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
        # cv2.imshow('f', img)
        # cv2.waitKey(1)
        writer.write(img)
        f += 2
        pbar.update(1)
    video.release()
    writer.release()
    pbar.close()


def gt_video(index, name_sequence):
    dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))
    Kinv = dataset['cam_K'][index].inverse()

    output_video_name = os.path.join(name_sequence + '_gt_result.mp4')
    result_name = os.path.join(name_sequence + '_gt_result.pt')

    if os.path.exists(output_video_name):
        return 0

    video_name = os.path.join(name_sequence + '_origin_video.mp4')

    pose_p, tran_p = torch.load(result_name)

    video = cv2.VideoCapture(video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    render = art.Renderer(resolution=(frame_width, frame_height), official_model_file=paths.smpl_file)
    f = 0

    pbar = tqdm.tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing Frames', dynamic_ncols=True)

    while True:
        im = video.read()[1]
        if im is None:
            break
        verts = \
            body_model.forward_kinematics(pose_p[0][f].view(-1, 24, 3, 3), tran=tran_p[0][f].view(-1, 3),
                                          calc_mesh=True)[
                2][0]
        img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
        cv2.imshow('f', img)
        cv2.waitKey(1)
        writer.write(img)
        f += 2
        pbar.update(1)
    video.release()
    writer.release()
    pbar.close()


if __name__ == '__main__':
    sequence_idx = 13

    name_sequence = evaluate_pw3d_ours(sequence_idx, occ=False)
    # generate_video(name_sequence)
    #
    # view(sequence_idx, name_sequence)
    # gt_video(sequence_idx, name_sequence)
