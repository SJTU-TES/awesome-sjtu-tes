import sys
thismodule = sys.modules[__name__]

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(123)


def get_autoencoder(config):
    ae_cls = getattr(thismodule, config.autoencoder.cls)
    return ae_cls(config.autoencoder)


class ConvEncoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        conv_pool = None if config.conv_pool is None else getattr(nn, config.conv_pool)
        encoder = cls(config.channels, config.padding, config.kernel_size, config.conv_stride, conv_pool)
        return encoder

    def __init__(self, channels, padding=3, kernel_size=8, conv_stride=2, conv_pool=None):
        super(ConvEncoder, self).__init__()

        self.in_channels = channels[0]

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 1

        for i in range(nr_layer):
            if conv_pool is None:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
            else:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
                model.append(conv_pool(kernel_size=2, stride=2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x[:, :self.in_channels, :]
        x = self.model(x)
        return x


class ConvDecoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        decoder = cls(config.channels, config.kernel_size)
        return decoder

    def __init__(self, channels, kernel_size=7):
        super(ConvDecoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)          # whether to add tanh a last?
                #model.append(nn.Dropout(p=0.2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.gan_type = config.gan_type
        encoder_cls = getattr(thismodule, config.encoder_cls)
        self.encoder = encoder_cls.build_from_config(config)
        self.linear = nn.Linear(config.channels[-1], 1)

    def forward(self, seqs):

        code_seq = self.encoder(seqs)
        logits = self.linear(code_seq.permute(0, 2, 1))
        return logits

    def calc_dis_loss(self, x_gen, x_real):

        fake_logits = self.forward(x_gen)
        real_logits = self.forward(x_real)

        if self.gan_type == 'lsgan':
            loss = torch.mean((fake_logits - 0) ** 2) + torch.mean((real_logits - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(fake_logits, requires_grad=False)
            all1 = torch.ones_like(real_logits, requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(fake_logits), all0) +
                              F.binary_cross_entropy(F.sigmoid(real_logits), all1))
        else:
            raise NotImplementedError

        return loss

    def calc_gen_loss(self, x_gen):

        logits = self.forward(x_gen)
        if self.gan_type == 'lsgan':
            loss = torch.mean((logits - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(logits, requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(logits), all1))
        else:
            raise NotImplementedError

        return loss


class Autoencoder3f(nn.Module):

    def __init__(self, config):
        super(Autoencoder3f, self).__init__()

        assert config.motion_encoder.channels[-1] + config.body_encoder.channels[-1] + \
               config.view_encoder.channels[-1] == config.decoder.channels[0]

        self.n_joints = config.decoder.channels[-1] // 3
        self.body_reference = config.body_reference

        motion_cls = getattr(thismodule, config.motion_encoder.cls)
        body_cls = getattr(thismodule, config.body_encoder.cls)
        view_cls = getattr(thismodule, config.view_encoder.cls)

        self.motion_encoder = motion_cls.build_from_config(config.motion_encoder)
        self.body_encoder = body_cls.build_from_config(config.body_encoder)
        self.view_encoder = view_cls.build_from_config(config.view_encoder)
        self.decoder = ConvDecoder.build_from_config(config.decoder)

        self.body_pool = getattr(F, config.body_encoder.global_pool) if config.body_encoder.global_pool is not None else None
        self.view_pool = getattr(F, config.view_encoder.global_pool) if config.view_encoder.global_pool is not None else None

    def forward(self, seqs):
        return self.reconstruct(seqs)

    def encode_motion(self, seqs):
        motion_code_seq = self.motion_encoder(seqs)
        return motion_code_seq

    def encode_body(self, seqs):
        body_code_seq = self.body_encoder(seqs)
        kernel_size = body_code_seq.size(-1)
        body_code = self.body_pool(body_code_seq, kernel_size)  if self.body_pool is not None else body_code_seq
        return body_code, body_code_seq

    def encode_view(self, seqs):
        view_code_seq = self.view_encoder(seqs)
        kernel_size = view_code_seq.size(-1)
        view_code = self.view_pool(view_code_seq, kernel_size)  if self.view_pool is not None else view_code_seq
        return view_code, view_code_seq

    def decode(self, motion_code, body_code, view_code):
        if body_code.size(-1) == 1:
            body_code = body_code.repeat(1, 1, motion_code.shape[-1])
        if view_code.size(-1) == 1:
            view_code = view_code.repeat(1, 1, motion_code.shape[-1])
        complete_code = torch.cat([motion_code, body_code, view_code], dim=1)
        out = self.decoder(complete_code)
        return out

    def cross3d(self, x_a, x_b, x_c):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_b)
        view_c, _ = self.encode_view(x_c)
        out = self.decode(motion_a, body_b, view_c)
        return out

    def cross2d(self, x_a, x_b, x_c):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_b)
        view_c, _ = self.encode_view(x_c)
        out = self.decode(motion_a, body_b, view_c)
        batch_size, channels, seq_len = out.size()
        n_joints = channels // 3
        out = out.view(batch_size, n_joints, 3, seq_len)
        out = out[:, :, [0, 2], :]
        out = out.view(batch_size, n_joints * 2, seq_len)
        return out
    
    def cross2d_adv(self, x_a, x_b, x_c):
        x_a.cpu()
        x_a_shape = x_a.shape
        print(x_a.shape)
        #motion_a_org = self.encode_motion(x_a)
        print(x_a)


        # The heatmap image is saved as 'tensor_heatmap.png' in the current directory

        # for i in range(0,119):
        #     x_a[0][11][i]+=1

        #x_a[0][7][60]+=0.01

        #motion_a = self.encode_motion(x_a)
        # print(motion_a.shape)
        # print(motion_a[0][0]-motion_a_org[0][0])
        # res = motion_a[0] - motion_a_org[0]
        # res = res.cpu().detach().numpy()
        # # Code for plotting the heatmap
        # plt.figure(figsize=(15, 10))
        # plt.imshow(res, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title('Heatmap of the Tensor')

        # # Save the heatmap to a local file
        # plt.savefig('/home/fazhong/studio/transmomo.pytorch/tensor_heatmap2.png')
        # plt.close()

        initial_motion_a = self.encode_motion(x_a)  # 计算初始的motion_a

        # 定义一个函数来计算motion的变化量
        def motion_change(motion_a, initial_motion_a):
            return (motion_a - initial_motion_a).norm()

        # 设置初始的最大变化量为0
        max_change = 0

        # 扰动次数，可以根据需要更改
        num_perturbations = 10000
        init_a = x_a.clone()
        for _ in range(num_perturbations):
            # 复制x_a以避免在原始数据上修改
            x_a_perturbed = x_a.clone().cpu()

            # 选择要扰动的随机点
            batch_idx, seq_idx, feature_idx = (torch.randint(0, x_a.size(0), (1,)),
                                            torch.randint(0, x_a.size(1), (1,)),
                                            torch.randint(0, x_a.size(2), (1,)))

            # 在选定点上加上扰动
            x_a_perturbed[batch_idx, seq_idx, feature_idx] += 10 * torch.randn(1)

            # 计算扰动后的motion_a
            perturbed_motion_a = self.encode_motion(x_a_perturbed.to('cuda:0'))

            # 计算变化量
            change = motion_change(perturbed_motion_a, initial_motion_a)

            # 如果变化量大于之前保存的最大变化量，则更新x_a和最大变化量
            if change > max_change:
                x_a = x_a_perturbed
                max_change = change

        # 最后，x_a将是导致最大motion_a变化的扰动版本
        # max_change是这个变化量
        # print(max_change)
        # print(max_change.shape)

        print(x_a_perturbed - init_a.cpu())
        motion_a = self.encode_motion(x_a_perturbed.to('cuda:0'))
        # motion_a = self.encode_motion(x_a.to('cuda:0'))
        body_b, _ = self.encode_body(x_b)
        view_c, _ = self.encode_view(x_c)

        out = self.decode(motion_a, body_b, view_c)
        batch_size, channels, seq_len = out.size()
        n_joints = channels // 3
        out = out.view(batch_size, n_joints, 3, seq_len)
        out = out[:, :, [0, 2], :]
        out = out.view(batch_size, n_joints * 2, seq_len)
        return out
    
    def cross2d_one(self, x_a):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_a)
        view_c, _ = self.encode_view(x_a)
        
        out = self.decode(motion_a, body_b, view_c)
        batch_size, channels, seq_len = out.size()
        n_joints = channels // 3
        out = out.view(batch_size, n_joints, 3, seq_len)
        out = out[:, :, [0, 2], :]
        out = out.view(batch_size, n_joints * 2, seq_len)
        return out

    def adv_cross(self,x_a):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_a)
        view_c, _ = self.encode_view(x_a)
        return motion_a

    def reconstruct3d(self, x):
        motion_code = self.encode_motion(x)
        body_code, _ = self.encode_body(x)
        view_code, _ = self.encode_view(x)
        out = self.decode(motion_code, body_code, view_code)
        return out

    def reconstruct2d(self, x):
        motion_code = self.encode_motion(x)
        body_code, _ = self.encode_body(x)
        view_code, _ = self.encode_view(x)
        out = self.decode(motion_code, body_code, view_code)
        batch_size, channels, seq_len = out.size()
        n_joints = channels // 3
        out = out.view(batch_size, n_joints, 3, seq_len)
        out = out[:, :, [0, 2], :]
        out = out.view(batch_size, n_joints * 2, seq_len)
        return out

    def interpolate(self, x_a, x_b, N):

        step_size = 1. / (N-1)
        batch_size, _, seq_len = x_a.size()

        motion_a = self.encode_motion(x_a)
        body_a, body_a_seq = self.encode_body(x_a)
        view_a, view_a_seq = self.encode_view(x_a)

        motion_b = self.encode_motion(x_b)
        body_b, body_b_seq = self.encode_body(x_b)
        view_b, view_b_seq = self.encode_view(x_b)

        batch_out = torch.zeros([batch_size, N, N, 2 * self.n_joints, seq_len])

        for i in range(N):
            motion_weight = i * step_size
            for j in range(N):
                body_weight = j * step_size
                motion = (1. - motion_weight) * motion_a + motion_weight * motion_b
                body = (1. - body_weight) * body_a + body_weight * body_b
                view = (1. - body_weight) * view_a + body_weight * view_b
                out = self.decode(motion, body, view)
                batch_size, channels, seq_len = out.size()
                n_joints = channels // 3
                out = out.view(batch_size, n_joints, 3, seq_len)
                out = out[:, :, [0, 2], :]
                out = out.view(batch_size, n_joints * 2, seq_len)
                batch_out[:, i, j, :, :] = out

        return batch_out


