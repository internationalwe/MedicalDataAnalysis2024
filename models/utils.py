import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models


def get_model(model_str: str):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """
    if model_str == 'eifficientnet_b3':
        return Efficientnet_b3
    elif model_str == 'eifficientnet_b4':
        return Efficientnet_b4
    elif model_str == 'eifficientnet_b5':
        return Efficientnet_b5
    elif model_str == 'eifficientnet_b6':
        return Efficientnet_b6
    elif model_str == 'convnext_base':
        return ConvNext_base
    elif model_str == 'convnext_large':
        return ConvNext_large 
    elif model_str == 'vit_b_16':
        return ViT_B_16  
    elif model_str == 'mobilenet_v3_small':
        return Mobilenet_v3_S     
    elif model_str == 'mobilenet_v3_large':
        return Mobilenet_v3_L     
    elif model_str == 'mobileone_s4':
        return Mobileone_s4
    elif model_str == 'coatnet_0_rw_224':
        return CoAtnet_rw_224 
    elif model_str == 'tinynet_c':
        return TinyNet_c  
    elif model_str == 'tinynet_e':
        return TinyNet_E  
    elif model_str == 'tiny_vit':
        return tiny_vit_21m_224_dist_in22k_ft_in1k
    elif model_str == 'tf_efficientnet_b5.ns_jft_in1k':
        return tf_efficientnet_b5_ns
    elif model_str == 'tf_efficientnetv2_m.in21k_ft_in1k':
        return tf_efficientnetv2_m_in21k
    elif model_str == 'caformer_b36.sail_in22k_ft_in1k':
        return caformer_b36_sail_in22k_ft_in1k
    elif model_str == 'caformer_s36.sail_in22k_ft_in1k':
        return caformer_s36_sail_in22k_ft_in1k
    elif model_str == 'convformer_m36.sail_in22k_ft_in1k':
        return convformer_m36_sail_in22k_ft_in1k
    elif model_str == 'tiny_vit_21m_224.dist_in22k_ft_in1k':
        return tiny_vit_21m_224_dist_in22k_ft_in1k
    elif model_str == 'convnext_small.in12k_ft_in1k':
        return convnext_small_in12k_ft_in1k
    elif model_str == 'tf_efficientnet_b2.ns_jft_in1k':
        return tf_efficientnet_b2_ns_jft_in1k
    elif model_str == 'tf_efficientnet_b2.ns_jft_in1k_froze':
        return tf_efficientnet_b2_ns_jft_in1k_froze
    elif model_str == 'tf_efficientnet_b3.ns_jft_in1k':
        return tf_efficientnet_b3_ns_jft_in1k
    elif model_str == 'efficientformerv2_l.snap_dist_in1k':
        return efficientformerv2_l_snap_dist_in1k
    elif model_str == 'efficientvit_b3.r224_in1k':
        return efficientvit_b3_r224_in1k
    elif model_str == 'tiny_vit_11m_224.dist_in22k_ft_in1k':
        return tiny_vit_11m_224_dist_in22k_ft_in1k
    elif model_str == 'maxvit_tiny_rw_224.sw_in1k':
        return maxvit_tiny_rw_224_sw_in1k
    elif model_str == 'convnextv2_tiny.fcmae_ft_in22k_in1k':
        return convnextv2_tiny_fcmae_ft_in22k_in1k
    elif model_str == 'tiny_vit_21m_224_dist_in22k_ft_in1k_froze':
        return tiny_vit_21m_224_dist_in22k_ft_in1k_froze
    elif model_str == 'tiny_vit_21m_224_dist_in22k_ft_in1k_froze012':
        return tiny_vit_21m_224_dist_in22k_ft_in1k_froze012
    elif model_str == 'caformer_b36_sail_in22k_ft_in1k_froze01':
        return caformer_b36_sail_in22k_ft_in1k_froze01
    elif model_str == 'convformer_m36_sail_in22k_ft_in1k_freeze':
        return convformer_m36_sail_in22k_ft_in1k_freeze
    elif model_str == "deit3_small_patch16_384.fb_in22k_ft_in1k":
        return deit3_small_patch16_384_fb_in22k_ft_in1k
    elif model_str == "caformer_s36.sail_in22k_ft_in1k_384":
        return caformer_s36_sail_in22k_ft_in1k_384
    elif model_str == "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k":
        return swinv2_base_window12to16_192to256_ms_in22k_ft_in1k
    elif model_str =="maxvit_nano_rw_256.sw_in1k":
        return maxvit_nano_rw_256_sw_in1k
    elif model_str == "deit3_small_patch16_224.fb_in22k_ft_in1k":
        return deit3_small_patch16_224_fb_in22k_ft_in1k
    elif model_str == "convnext_tiny.in12k_ft_in1k":
        return convnext_tiny_in12k_ft_in1k
    # elif model_str == 'xcit_small_12_p8_224.fb_dist_in1k':
    #     return timm.create_model(model_str,pretrained=True)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    

    

class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b4, self).__init__()
        
        self.model = models.efficientnet_b4(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x


class Efficientnet_b3(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b3, self).__init__()
        
        self.model = models.efficientnet_b3(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x
    
class Efficientnet_b5(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b5, self).__init__()
        
        self.model = models.efficientnet_b5(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x

class Efficientnet_b6(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b6, self).__init__()
        
        self.model = models.efficientnet_b6(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x


class ConvNext_base(nn.Module):
    def __init__(self, num_classes):
        super(ConvNext_base, self).__init__()
        
        self.model = models.convnext_base(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.LayerNorm([pre_layer]),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)        
        x = self.linear(x)
        return x


class ConvNext_large(nn.Module):
    def __init__(self, num_classes):
        super(ConvNext_large, self).__init__()
        
        self.model = models.convnext_large(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.LayerNorm([pre_layer]),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)        
        x = self.linear(x)
        return x

class ViT_B_16(nn.Module):
    def __init__(self, num_classes):
        super(ViT_B_16, self).__init__()
        
        self.model = models.vit_b_16(pretrained=True)
        pre_layer = self.model.heads[-1].in_features
        self.model.heads = Identity()

        self.linear = nn.Sequential(
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    

class Mobilenet_v3_L(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v3_L, self).__init__()
        
        self.model = models.mobilenet_v3_large(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier[-1] = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class Mobilenet_v3_S(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v3_S, self).__init__()
        
        self.model = models.mobilenet_v3_small(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier[-1] = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x 


class Mobileone_s4(nn.Module):
    def __init__(self, num_classes):
        super(Mobileone_s4, self).__init__()
        
        self.model = timm.create_model("mobileone_s4", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    

class CoAtnet_rw_224(nn.Module):
    def __init__(self, num_classes):
        super(CoAtnet_rw_224, self).__init__()
        
        self.model = timm.create_model("coatnet_0_rw_224", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class TinyNet_c(nn.Module):
    def __init__(self, num_classes):
        super(TinyNet_c, self).__init__()
        
        self.model = timm.create_model("tinynet_c", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class TinyNet_E(nn.Module):
    def __init__(self, num_classes):
        super(TinyNet_E, self).__init__()
        
        self.model = timm.create_model("tinynet_e", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnet_b5_ns(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b5_ns, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b5.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnetv2_m_in21k(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnetv2_m_in21k, self).__init__()
        
        self.model = timm.create_model("tf_efficientnetv2_m.in21k_ft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnet_b2_ns_jft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b2_ns_jft_in1k, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b2.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(pre_layer, num_classes),
                                              nn.Softmax())
        # self.model.classifier = nn.Linear(pre_layer, num_classes)

        
    def forward(self, x):
        x = self.model(x)
        # x = F.softmax(x, dim=1)
        return x
   
class tf_efficientnet_b2_ns_jft_in1k_froze(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b2_ns_jft_in1k_froze, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b2.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(pre_layer, num_classes),
                                              nn.Softmax())
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.blocks[4].parameters():
            param.requires_grad = True
        for param in self.model.blocks[5].parameters():
            param.requires_grad = True
        for param in self.model.blocks[6].parameters():
            param.requires_grad = True
        
        self.model.conv_head.requires_grad = True
        self.model.global_pool.require_grad = True
        self.model.bn2.require_grad = True
        self.model.classifier.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        # x = F.softmax(x, dim=1)
        return x    
class tf_efficientnet_b3_ns_jft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b3_ns_jft_in1k, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(pre_layer, num_classes),
                                              nn.Softmax())
        # self.model.classifier = nn.Linear(pre_layer, num_classes)

        
    def forward(self, x):
        x = self.model(x)
        # x = F.softmax(x, dim=1)
        return x
    
class caformer_b36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(caformer_b36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("caformer_b36.sail_in22k_ft_in1k", pretrained=True,drop_rate=0.2)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x

    
class caformer_s36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(caformer_s36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("caformer_s36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
class caformer_s36_sail_in22k_ft_in1k_384(nn.Module):
    def __init__(self, num_classes):
        super(caformer_s36_sail_in22k_ft_in1k_384, self).__init__()
        
        self.model = timm.create_model("caformer_s36.sail_in22k_ft_in1k_384", pretrained=False,num_classes=num_classes)
        # pre_layer = self.model.head.fc.fc2.in_features
        # # self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        # self.model.head.fc.fc2 = Identity()
        # self.linear =  nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        # x = self.linear(x)
        return x
    
class convformer_m36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convformer_m36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("convformer_m36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class deit3_small_patch16_384_fb_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(deit3_small_patch16_384_fb_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("deit3_small_patch16_384.fb_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.in_features
        self.model.head = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class convnext_small_in12k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convnext_small_in12k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("convnext_small.in12k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class tiny_vit_21m_224_dist_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tiny_vit_21m_224_dist_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x
    
class tiny_vit_11m_224_dist_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tiny_vit_11m_224_dist_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("tiny_vit_11m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class efficientformerv2_l_snap_dist_in1k(nn.Module):
    def __init__(self, num_classes):
        super(efficientformerv2_l_snap_dist_in1k, self).__init__()
        
        self.model = timm.create_model("efficientformerv2_l.snap_dist_in1k", pretrained=True)
        pre_layer = self.model.head.in_features
        self.model.head = nn.Linear(pre_layer, num_classes)
        self.model.head_dist = nn.Linear(pre_layer, num_classes)

        
    def forward(self, x):
        x = self.model(x)
        return x
    
class efficientvit_b3_r224_in1k(nn.Module):
    def __init__(self, num_classes):
        super(efficientvit_b3_r224_in1k, self).__init__()
        
        self.model = timm.create_model("efficientvit_b3.r224_in1k", pretrained=True)
        pre_layer = self.model.head.classifier[4].in_features
        self.model.head.classifier[4] = nn.Linear(pre_layer, num_classes)


        
    def forward(self, x):
        x = self.model(x)
        return x
    
class efficientformerv2_l_snap_dist_in1k(nn.Module):
    def __init__(self, num_classes):
        super(efficientformerv2_l_snap_dist_in1k, self).__init__()
        
        self.model = timm.create_model("efficientformerv2_l.snap_dist_in1k", pretrained=True)
        pre_layer = self.model.head.in_features
        self.model.head = nn.Linear(pre_layer, num_classes)
        self.model.head_dist = nn.Linear(pre_layer, num_classes)

        
    def forward(self, x):
        x = self.model(x)
        return x
    
class maxvit_tiny_rw_224_sw_in1k(nn.Module):
    def __init__(self, num_classes):
        super(maxvit_tiny_rw_224_sw_in1k, self).__init__()
        
        self.model = timm.create_model("maxvit_tiny_rw_224.sw_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)


        
    def forward(self, x):
        x = self.model(x)
        return x
    
class convnextv2_tiny_fcmae_ft_in22k_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convnextv2_tiny_fcmae_ft_in22k_in1k, self).__init__()
        
        self.model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)


        
    def forward(self, x):
        x = self.model(x)
        return x
    
class TinyVit(nn.Module):
    def __init__(self, num_class):
        super(TinyVit, self).__init__()
        
        self.model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.head.fc = nn.Linear(pre_layer, num_class)

    def forward(self, x):
        
        x = self.model(x)

        return x
class swinv2_base_window12to16_192to256_ms_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(swinv2_base_window12to16_192to256_ms_in22k_ft_in1k, self).__init__()
        self.model = timm.create_model("swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True,num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class tiny_vit_21m_224_dist_in22k_ft_in1k_froze(nn.Module):
    def __init__(self, num_classes):
        super(tiny_vit_21m_224_dist_in22k_ft_in1k_froze, self).__init__()
        
        self.model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.stages[2].parameters():
            param.requires_grad = True
        for param in self.model.stages[3].parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x
class tiny_vit_21m_224_dist_in22k_ft_in1k_froze012(nn.Module):
    def __init__(self, num_classes):
        super(tiny_vit_21m_224_dist_in22k_ft_in1k_froze012, self).__init__()
        
        self.model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        # for param in self.model.stages[2].parameters():
        #     param.requires_grad = True
        for param in self.model.stages[3].parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x

    
class caformer_b36_sail_in22k_ft_in1k_froze01(nn.Module):
    def __init__(self, num_classes):
        super(caformer_b36_sail_in22k_ft_in1k_froze01, self).__init__()
        
        self.model = timm.create_model("caformer_b36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.stages[2].parameters():
            param.requires_grad = True
        for param in self.model.stages[3].parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class convformer_m36_sail_in22k_ft_in1k_freeze(nn.Module):
    def __init__(self, num_classes):
        super(convformer_m36_sail_in22k_ft_in1k_freeze, self).__init__()
        
        self.model = timm.create_model("convformer_m36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.stages[2].parameters():
            param.requires_grad = True
        for param in self.model.stages[3].parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x
class maxvit_nano_rw_256_sw_in1k(nn.Module):
    def __init__(self, num_classes):
        super(maxvit_nano_rw_256_sw_in1k, self).__init__()
        self.model = timm.create_model("maxvit_nano_rw_256.sw_in1k", pretrained=True,num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
class deit3_small_patch16_224_fb_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(deit3_small_patch16_224_fb_in22k_ft_in1k, self).__init__()
        self.model = timm.create_model("deit3_small_patch16_224.fb_in22k_ft_in1k", pretrained=True,num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
class convnext_tiny_in12k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convnext_tiny_in12k_ft_in1k, self).__init__()
        self.model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True,num_classes = num_classes)

    def forward(self, x):
        x = self.model(x)
        return x