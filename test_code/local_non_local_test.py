import torch
import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2014)

class LocalNonLocal2D(nn.Module):

    def __init__(self,
                 kernel_size):
        super(LocalNonLocal2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def embedded_gaussian(self, theta_x, phi_x):
        """

        :param theta_x: [N, C, H, W]
        :param phi_x: [N, C, H, W]
        :return:  [N, K * K , H * W]
        """
        N, C, H, W = theta_x.shape
        # [N, C*K*K, H*W]，第二个维度前K*K个为c=0的特征，依此类推。先C后K
        K = self.kernel_size
        unfold_phi = torch.nn.functional.unfold(phi_x,
                                                self.kernel_size,
                                                padding=self.padding,
                                                stride=1)
        # phi: [N, K*K*H*W, C]
        phi = unfold_phi.reshape(N, C, K * K, H * W) \
            .permute(0, 2, 3, 1).reshape(N, K * K * H * W, C)
        # theta: [N, K*K*H*W, C]
        theta = theta_x.reshape(N, C, H * W).permute(0, 2, 1)
        theta = torch.cat([theta for i in range(K * K)], dim=1)

        # weight: [N,  K*K*H*W]
        weight = (phi * theta).sum(-1)

        # [N, K * K , H * W]
        pairwise_weight = weight.reshape(N, K * K, H * W).permute(0, 2, 1).reshape(N * H * W, K * K)
        pairwise_weight = pairwise_weight / torch.sum(pairwise_weight, dim=1).unsqueeze(-1)

        # pairwise_weight = pairwise_weight.softmax(dim=1)
        # v, topk_ind = torch.topk(pairwise_weight, min(5, K * K), dim=1)
        # topk_mask = weight.new_zeros(N * H * W, K * K)
        # topk_mask = topk_mask.scatter_(1, topk_ind, 1).float()
        # pairwise_weight = pairwise_weight * topk_mask
        # norm = torch.sum(v, dim=1).unsqueeze(-1)
        # pairwise_weight = pairwise_weight / norm
        pairwise_weight = pairwise_weight.reshape(N, H * W, K * K).permute(0, 2, 1)

        return pairwise_weight

    # x for calculate weight, feat for reweighted
    def forward(self, x, feat_in):
        n, _, h, w = x.shape

        # feat: N x C x H x W
        # unfold_feat: [N, C*K*K, H*W]
        N, C, H, W = feat_in.shape
        K = self.kernel_size
        unfold_feat = torch.nn.functional.unfold(feat_in,
                                                 self.kernel_size,
                                                 padding=self.padding,
                                                 stride=1)
        # (N, C, K*K, H*W)
        feat = unfold_feat.reshape(N, C, K * K, H * W)
        # (N*H*W, K*K, C)
        feat = feat.permute(0, 3, 2, 1).reshape(N * H * W, K * K, C)

        # theta_x: [N, C, H, W]
        theta_x = x

        # phi_x: [N, C, H, W]
        phi_x = x

        # pairwise_weight:  [N, K*K , H*W]
        pairwise_weight = self.embedded_gaussian(theta_x, phi_x)
        #  [N*H*W, 1, K*K]
        pairwise_weight = pairwise_weight.permute(0, 2, 1).reshape(N * H * W, 1, K * K)

        # pairwise_weight: [N*H*W, 1, K*K], feat: (N*H*W, K*K, C)
        output = torch.matmul(pairwise_weight, feat)  # (N*H*W, 1, C)

        output = output.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return output, pairwise_weight

if __name__ == '__main__':
    LNL = LocalNonLocal2D(3)

    a = torch.rand([2, 1000000, 3, 3]) - 0.5
    o, w = LNL(a, a)
    # print(a, o, w)
    print(torch.abs(o - a))







