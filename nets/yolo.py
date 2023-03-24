import torch
import torch.nn as nn

from .attention import cbam_block
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


#  每个head在加强提取特征时的残差块，每个head由六个残差块构成
class Attention_ResBlock(nn.Module):
    def __init__(self, input_channel, depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.conv_block = Conv(in_channels=input_channel, out_channels=input_channel, ksize=3, stride=1, act=act)
        self.attention_eca = cbam_block(input_channel)
        self.act1 = nn.Tanh()

    def forward(self, x):
        return self.act1(self.attention_eca(self.conv_block(x)) + x)


# 通过共享的特征层，每个子任务利用注意力机制加权提取自己所需要的特征
# 此处的attention_mask是学习得到的，不是由自注意力机制通过自身生成的
class Specific_Attention_Block(nn.Module):
    def __init__(self, input_channel, depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.conv_block = Conv(in_channels=input_channel, out_channels=input_channel, ksize=3, stride=1, act=act)
        self.act1 = nn.Tanh()

    def forward(self, input):
        x, y = input
        return self.conv_block(x) * y


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.back_feature = nn.ModuleList()
        self.force_feature = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append \
                (Conv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                      act=act))
            self.back_feature.append(
                nn.ModuleList([Attention_ResBlock(input_channel=int(256 * width), depthwise=False, act=act)
                               for _ in range(6)]))
            temp_conv = nn.ModuleList()
            for j in range(2):
                temp_conv.append(
                    nn.ModuleList([Specific_Attention_Block(input_channel=int(256 * width)) for _ in range(3)]))
            self.force_feature.append(temp_conv)
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        inputs = [inputs[item] for item in ("dark3", "dark4", "dark5")]
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#

        outputs = []
        for k, x in enumerate(inputs):
            print(x.shape)
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------
            x = self.stems[k](x)
            print(x.shape)
            # 加强特征提取，并保留每一步残差的结果
            conv_res = []
            for item in self.back_feature[k]:
                x = item.forward(x)
                conv_res.append(x)
            # 两个回归任务头
            attention_res = [[0] * 3] * 2
            for i in range(2):
                for j in range(3):
                    if j == 0:
                        attention_res[i][j] = self.force_feature[k][i][j]([conv_res[0], conv_res[1]])
                    elif j == 1:
                        attention_res[i][j] = self.force_feature[k][i][j](
                            [conv_res[2 * j] + attention_res[i][j - 1], conv_res[2 * j + 1]])
                    else:
                        attention_res[i][j] = self.force_feature[k][i][j](
                            [conv_res[2 * j] + attention_res[i][j - 1], conv_res[2 * j + 1]])

            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](attention_res[0][-1])
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](attention_res[1][-1])
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](attention_res[1][-1])
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_w = 0.5

        self.p3_p5 = CSPLayer(
            int(in_channels[2] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.p5_p3_1_conv = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.p5_p3_2_conv = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.p5_p3 = CSPLayer(
            int(in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        [feat1, feat2, feat3] = [input[f] for f in self.in_features]

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat3)
        # -------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        # -------------------------------------------#
        p3_p5 = self.upsample(self.upsample(feat3))
        p3_p5 = self.p3_p5(p3_p5)

        P5_upsample = self.upsample(P5)
        # -------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, self.conv_w * feat2], 1)
        # P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        # P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P4_upsample = torch.cat([P4_upsample, self.conv_w * feat1 + (1 - self.conv_w) * p3_p5], 1)

        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        P3_out = self.C3_p3(P4_upsample)
        p5_p3 = self.p5_p3_1_conv(P3_out)
        p5_p3 = self.p5_p3_2_conv(p5_p3)
        p5_p3 = self.p5_p3(p5_p3)
        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        P3_downsample = self.bu_conv2(P3_out)
        # -------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 512
        # -------------------------------------------#
        P4_out = self.C3_n3(P3_downsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        P4_downsample = self.bu_conv1(P4_out)
        # -------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        # -------------------------------------------#
        # P4_downsample = torch.cat([P4_downsample, P5 ], 1)
        P4_downsample = torch.cat([P4_downsample, self.conv_w * P5 + (1 - self.conv_w) * p5_p3], 1)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        P5_out = self.C3_n4(P4_downsample)
        output = {"dark3": P3_out, "dark4": P4_out, "dark5": P5_out}
        return output


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }

        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        self.backbone = CSPDarknet(depth, width, depthwise=False, act="silu")
        self.neck = nn.Sequential(*[YOLOPAFPN(depth, width, depthwise=True) for _ in range(2)])
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        backbone_outs = self.backbone.forward(x)
        neck_outs = self.neck.forward(backbone_outs)
        outputs = self.head.forward(neck_outs)
        return outputs

# if __name__ == '__main__':
#     print("hello world")
#     testModel = YoloBody(10, 'm')
#     # device = torch.device("cuda")
#     # testModel = testModel.to('cuda')
#     print("hello world")
#     # print(testModel)
#     imageDate = torch.ones([2, 3, 640, 640])
#     # imageDate = imageDate.to('cuda')
#     print(imageDate.shape)
#     targets = testModel(imageDate)
#     print(len(targets))
#     print(targets[0].shape)
#     print(targets[1].shape)
#     print(targets[2].shape)
