import functools
import numpy as np
import spconv
import torch
import torch.nn as nn
import pointpillars
import DDRnet
import regSeg
import fusion
import kornia
import fusion_for_three


class PillarFeatureNetMultiStep(nn.Module):
    def __init__(self):
        super(PillarFeatureNetMultiStep, self).__init__()
        self.PillarFeatureNet = pointpillars.PillarFeatureNet( num_input_features = 4,
                use_norm = False,
                num_filters=[128],
                with_distance=False,
                voxel_size=[0.25, 0.25, 6],
                pc_range=[-20.25, 0, -3, 20.25, 40.5, 3])


    def forward(self, voxels, num_points, coors):
        # features: T x [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: T x [concated_num_points]
        # returns list of T x [num_voxels]
        self.eval()
        with torch.no_grad():
          t = len(voxels)
          output = []
          for i in range(t):
              voxels_single = voxels[i]
              num_points_i = num_points[i]
              coors_i = coors[i]
              coors_i = coors_i.int()

              output_i=self.PillarFeatureNet(voxels_single,num_points_i,coors_i)
              output.append(output_i.contiguous())
          return output

class PointPillarsScatter(nn.Module):
    """
    No gradients!
    """
    def __init__(self):
        super(PointPillarsScatter, self).__init__()
        self.middle_feature_extractor = pointpillars.PointPillarsScatter(output_shape=[1,1,162,162,128],num_input_features=128)
    def forward(self, voxel_features, coors, batch_size):
            output = self.middle_feature_extractor(voxel_features, coors, batch_size)
            return output

class PointPillarsScatterMultiStep(PointPillarsScatter):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(PointPillarsScatterMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
          if True:
              t = len(voxel_features)
              output = []
              for i in range(t):
                  voxel_features_i = voxel_features[i]
                  coors_i = coors[i]
                  coors_i = coors_i.int()
                  ret = self.middle_feature_extractor(voxel_features_i, coors_i, batch_size)
                # ret = ret.dense()
                #output.append(ret.detach())
                  output.append(ret)
              return output




class PillarFeatureNet(nn.Module):
    def __init__(self):
        super(PillarFeatureNet, self).__init__()
        self.PillarFeatureNet = pointpillars.PillarFeatureNet( num_input_features = 4,
                use_norm = False,
                num_filters=[128],
                with_distance=False,
                voxel_size=[0.25, 0.25, 10],
                pc_range=[-20.25, 0, -5, 20.25, 40.5, 5])

    def forward(self, voxels, num_points, coors):
        # features: T x [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: T x [concated_num_points]
        # returns list of T x [num_voxels]
        output=self.PillarFeatureNet(voxels,num_points,coors)
        return output

class DDRNet_forsegment(nn.Module):
    def __init__(self,
                  num_input_feature = 64,
                  num_classes = 5):
        super(DDRNet_forsegment, self).__init__()
        self.DDRNet = DDRnet.DDRNet(num_input_feature=num_input_feature, num_classes=5)

    def forward(self, x):
        if (self.training):
          out,out1 = self.DDRNet(x)
          ret_dict = {
              "bev_preds": out,
              "bev_preds1":out1,
          }
        else:
           out=self.DDRNet(x)
           ret_dict = {
              "bev_preds": out,
          }          
        return ret_dict
class Traver_Completion(nn.Module):
    def __init__(self,
                 num_classes=5,
                 num_input_feature=64):
        super(Traver_Completion, self).__init__()
        self.regseg = regSeg.RegSeg("exp48_decoder26",num_classes=num_classes,num_input_feature=num_input_feature)

    def forward(self, x, *args, **kwargs):
        out = self.regseg(x)
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class Traver_Completion_Fusion(nn.Module):
    def __init__(self,num_classes=5,
                 num_input_feature=192,
                 aggregation_type='pre', 
                 **kwargs
                 ):
        super(RegSeg_with_fussion, self).__init__(**kwargs)

        assert aggregation_type in ['pre', 'post', 'none'], aggregation_type
        self.aggregation_type = aggregation_type

        if aggregation_type != 'none':
            self.prefusion=fusion.Fusion_Module(output_channel=192)

            self.regseg = regSeg.RegSeg("exp48_decoder26", 5,num_input_feature=192)
            def get_poses(input_pose):
                mat = torch.zeros(input_pose.shape[0], # batch_size
                                  input_pose.shape[1], # t
                                  3, 3, dtype=input_pose.dtype,
                                  device=input_pose.device)

                mat[:, :, 0] = input_pose[:, :, :3]
                mat[:, :, 1] = input_pose[:, :, 3:6]
                mat[:, :, 2, 2] = 1.0

                return mat[:, :, None]

            self.get_poses = get_poses

    def forward(self, x, seq_start=None, input_pose=None):
        n, c, h, w = x[0].shape

        t = len(x)
        #b t c h w
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.view((-1,) + x.size()[2:])  # Fuse dim 0 and 1
        if self.aggregation_type != 'none':
            if seq_start is None:
                self.hidden_state = None
            else:
                # sanity check: only the first index can be True
                assert(torch.any(seq_start[1:]) == False)

                if seq_start[0]:  # start of a new sequence
                    self.hidden_state = None

        if self.aggregation_type == 'pre':
            pose=self.get_poses(input_pose[None])
            M1 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 0, 0])[:, :2]
            M2 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 2, 0])[:, :2]
            x1 =x[0,:,:,:].unsqueeze(0)
            x2=x[2,:,:,:].unsqueeze(0)

            #from matplotlib.pyplot import show, imshow, figure, imsave

            M1_t = kornia.geometry.warp_affine(x1, M1, dsize=[512,512],
                                                        align_corners=False)
            M2_t = kornia.geometry.warp_affine(x2, M2, dsize=[512,512],
                                                        align_corners=False)

            x_fusion=self.prefusion(M1_t,x[1,:,:,:].unsqueeze(0),192)


        out = self.regseg(x_fusion)
        if self.aggregation_type == 'post':
            pose_to_transform=self.get_poses(input_pose[None])
            #Pose1*T =Pose2 -> T =inv(pose1)*pose2
            M1 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 0, 0])[:, :2]
            M2 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 2, 0])[:, :2]
            M1_t = kornia.geometry.warp_affine(out[0,:,:,:], M1, dsize=[512,512],
                                                        align_corners=False)
            M2_t = kornia.geometry.warp_affine(out[2,:,:,:], M2, dsize=[512,512],
                                                        align_corners=False)
            x_fusion=self.postfusion(M1_t,M2_t,5)
            out=self.postfusion(x_fusion,out[1,:,:,:],5)

        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class Traver_Completion_Fusion_three (nn.Module):
    def __init__(self,num_classes=5,
                 num_input_feature=128,
                 aggregation_type='pre', 
                 **kwargs
                 ):
        super(Traver_Completion_Fusion , self).__init__(**kwargs)

        assert aggregation_type in ['pre', 'post', 'none'], aggregation_type
        self.aggregation_type = aggregation_type

        if aggregation_type != 'none':
            self.prefusion=fusion_for_three.Fusion_Module(output_channel=128)
            self.regseg = regSeg.RegSeg("exp48_decoder26", 5,num_input_feature=128)
            def get_poses(input_pose):
                # convert to matrix
                mat = torch.zeros(input_pose.shape[0], # batch_size
                                  input_pose.shape[1], # t
                                  3, 3, dtype=input_pose.dtype,
                                  device=input_pose.device)

                mat[:, :, 0] = input_pose[:, :, :3]
                mat[:, :, 1] = input_pose[:, :, 3:6]
                mat[:, :, 2, 2] = 1.0

                return mat[:, :, None]

            self.get_poses = get_poses

    def forward(self, x, seq_start=None, input_pose=None):
        n, c, h, w = x[0].shape
        t = len(x)
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.view((-1,) + x.size()[2:])  
        if self.aggregation_type != 'none':
            if seq_start is None:
                self.hidden_state = None
            else:
                # sanity check: only the first index can be True
                assert(torch.any(seq_start[1:]) == False)

                if seq_start[0]:  # start of a new sequence
                    self.hidden_state = None

        if self.aggregation_type == 'pre':
            pose=self.get_poses(input_pose[None])
            #Pose1*T =Pose2 -> T =inv(pose1)*pose2
            M1 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 0, 0])[:, :2]
            M2 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 2, 0])[:, :2]
            x1 =x[0,:,:,:].unsqueeze(0)
            x2=x[2,:,:,:].unsqueeze(0)

            M1_t = kornia.geometry.warp_affine(x1, M1, dsize=[512,512],
                                                        align_corners=False)
            M2_t = kornia.geometry.warp_affine(x2, M2, dsize=[512,512],
                                                        align_corners=False)

            x_fusion=self.prefusion(M1_t,x[1,:,:,:].unsqueeze(0),M2_t,128)

        out = self.regseg(x_fusion)
        if self.aggregation_type == 'post':
            pose_to_transform=self.get_poses(input_pose[None])
            #Pose1*T =Pose2 -> T =inv(pose1)*pose2
            M1 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 0, 0])[:, :2]
            M2 = torch.matmul(torch.inverse(pose[:, 1, 0]), pose[:, 2, 0])[:, :2]
            M1_t = kornia.geometry.warp_affine(out[0,:,:,:], M1, dsize=[512,512],
                                                        align_corners=False)
            M2_t = kornia.geometry.warp_affine(out[2,:,:,:], M2, dsize=[512,512],
                                                        align_corners=False)
            x_fusion=self.postfusion(M1_t,M2_t,5)
            out=self.postfusion(x_fusion,out[1,:,:,:],5)

        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict













