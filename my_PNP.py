import torch
from torch.utils.data import DataLoader
import articulate as art
import config
from articulate.utils.torch import *
from articulate.utils.print import *
from config import *
from tqdm import tqdm
import numpy as np
import utils
import os
from net.smplify.run import smplify_runner
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from distance_filtering import fitting_pnpresult


def cal_mpjpe(pose, gt_pose, cal_pampjpe=False):
    _, gt_joint_pos, gt_vertices = body_model.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, joint_pos, vertices = body_model.forward_kinematics(pose.cpu(), calc_mesh=True)
    keypoints_3d = torch.matmul(J_regressor_batch, vertices)[:, :14]
    pred_pelvis = keypoints_3d[:, [0], :].clone()
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    keypoints_3d = keypoints_3d - pred_pelvis
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    joint_connections = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8), (3, 6), (6, 9), (9, 12), (12, 15),
                         (12, 13), (13, 16), (12, 14), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
    frame_idx = 0

    ax.view_init(elev=-90, azim=-90)

    def update(frame_idx):
        ax.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        current_frame_data = joint_pos[frame_idx]

        for joint_data in current_frame_data:
            x, y, z = joint_data
            ax.scatter(x, y, z)  # 将z轴取反以使其朝屏幕内

        for connection in joint_connections:
            j1, j2 = connection
            x1, y1, z1 = current_frame_data[j1]
            x2, y2, z2 = current_frame_data[j2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'k')  # 使用'k'表示黑色线段

    animation = FuncAnimation(fig, update, frames=range(joint_pos.shape[0]), repeat=False)

    # 保存为60帧的视频
    # animation.save('blind_joint_animation.mp4', writer='ffmpeg', fps=60)

    plt.show()

    if cal_pampjpe:
        pampjpe = utils.reconstruction_error(keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        return torch.tensor(
            [(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean(),
             pampjpe.mean()])
    return torch.tensor(
        [(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean()])


def plot_trajectory(tran_p, tran_t):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    # 创建一个示例的tran_t ndarray，这里使用随机数据作为示例
    np.random.seed(0)
    tran_t = np.random.randn(2898, 3)

    # 创建一个绘图窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 调整视角，使x和y轴沿着屏幕方向，z轴朝屏幕内方向
    ax.view_init(elev=90, azim=-90)

    # 初始化一个空的三维线图
    line, = ax.plot([], [], [], lw=2)

    # 设置轴界限
    ax.set_xlim(min(tran_t[:, 0]), max(tran_t[:, 0]))
    ax.set_ylim(min(tran_t[:, 1]), max(tran_t[:, 1]))
    ax.set_zlim(min(tran_t[:, 2]), max(tran_t[:, 2]))

    # 更新函数，用于更新线图的数据
    def update(frame):
        line.set_data(tran_t[:frame, 0], tran_t[:frame, 1])
        line.set_3d_properties(tran_t[:frame, 2])
        return line,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(tran_t), blit=True)

    # 显示动画
    plt.show()


def myPNP(j2dc, pose, K):
    _, pose = body_model.forward_kinematics(pose.cpu(), calc_mesh=False)
    mapping_MP_SMPLJoint = [(23, 1), (24, 2), (25, 4), (25, 5), (27, 7), (28, 8), (31, 10), (32, 11), (11, 16),
                            (12, 17), (13, 18), (14, 19), (15, 20), (16, 21)]

    imagePoints, objectPoints = [], []

    # for mapping in mapping_MP_SMPLJoint:
    #     j1, j2 = mapping
    #     imagePoints.append(j2dc[..., j1, :2].numpy())
    #     objectPoints.append(pose[..., j2, :3].numpy())

    for i in range(len(j2dc)):
        points_2d, points_3d = [], []
        for mapping in mapping_MP_SMPLJoint:
            j1, j2 = mapping
            if j2dc[i, j1, -1] < 0.9:
                continue
            else:
                points_2d.append(j2dc[i, j1, :2].numpy())
                points_3d.append(pose[i, j2, :3].numpy())
        points_2d = np.array(points_2d)
        points_3d = np.array(points_3d)
        imagePoints.append(points_2d)
        objectPoints.append(points_3d)

    tran_pnp = []

    for i in range(len(j2dc)):
        if imagePoints[i].shape[0] < 4:
            tran_pnp.append(np.zeros((3, 1)))  # 可信点少于4，无法PNP
            continue
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints[i], imagePoints[i], K, None)
        tran_pnp.append(tvec)

    return tran_pnp


def show_path_pureIMU(tran):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    # 提取X、Y、Z坐标
    x = tran[:, 0]
    y = tran[:, 1]
    z = tran[:, 2]

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=-90)

    # 设置图形标题和轴标签
    ax.set_title('3D Point Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    # 创建一个初始线条对象
    line, = ax.plot(x[:1], y[:1], z[:1], color='blue', lw=2)

    # 更新函数，用于更新线条的数据和颜色
    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])

        # 通过改变颜色来表示轨迹
        color = (0, 0, frame / len(x))  # 逐渐变深的颜色
        line.set_color(color)

        return line,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=range(len(x)), blit=True, repeat=False, interval=5)

    # 显示动画
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_model = art.ParametricModel(paths.smpl_file)
    J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float()

    sequence_idx = 13

    dataset = torch.load(os.path.join('data/dataset_work/3DPW/', 'test' + '.pt'))
    data, label = [], []

    i = sequence_idx

    # test=dataset['joint2d_mp'][i][..., -1].numpy()

    print(dataset['name'][i])

    name_sequence = dataset['name'][i]
    result_name = os.path.join(name_sequence + '_result.pt')
    gt_result_name = os.path.join(name_sequence + '_gt_result.pt')

    K = dataset['cam_K'][i]
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

    if os.path.exists(result_name):
        pose_p, tran_p = torch.load(result_name)
        for d, l in test_dataloader:
            pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
            tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])

    pose_p = pose_p[0]
    tran_p = tran_p[0]
    pose_t = pose_t[0]
    tran_t = tran_t[0]

    # cal_mpjpe(pose_p, pose_t, cal_pampjpe=True)
    # plot_trajectory(tran_p.numpy(), tran_t.numpy())

    tran_pnp = myPNP(j2dc, pose_t, K.numpy())
    tran_p, tran_t = tran_p.numpy(), tran_t.numpy()
    tran_pnp = np.array(tran_pnp).squeeze()

    os.makedirs(name_sequence, exist_ok=True)

    np.save('./%s/pnp_distance.npy' % name_sequence, tran_pnp)
    np.save('./%s/gt_distance.npy' % name_sequence, tran_t)

    tran_pnp[:, 2] = fitting_pnpresult(tran_pnp, tran_t)

    distance = []

    for i in range(len(tran_t)):
        distance.append([tran_p[i, 2], tran_pnp[i, 2], tran_t[i, 2]])
    distance = np.array(distance)

    # 示例的distance数组（用随机数据代替）
    # distance = np.random.rand(2898, 3)

    pure_IMU_pose, pure_IMU_tran = torch.load('./%s/pure_IMU_result.pt' % name_sequence)
    pure_IMU_tran = pure_IMU_tran[0].numpy()

    # show_path_pureIMU(pure_IMU_tran)

    plt.figure(figsize=(16, 9))
    plt.scatter(range(distance.shape[0]), distance[:, 0], label='tran_p', color='red')
    plt.scatter(range(distance.shape[0]), distance[:, 1], label='tran_filteredPNP', color='green')
    plt.scatter(range(distance.shape[0]), distance[:, 2], s=5, label='tran_gt', color='blue')

    # 设置横轴标签和图例
    plt.xlabel('Index')
    plt.legend()

    # # 限制纵坐标范围在0到4
    # plt.ylim(0, max(distance[:, 2]) + 1)

    # 显示图例
    plt.show()
