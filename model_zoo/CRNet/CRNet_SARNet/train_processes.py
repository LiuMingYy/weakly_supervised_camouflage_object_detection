import numpy as np
import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps

criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()  # 忽略类别255
loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
    
def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:  # 翻转
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:  # 平移
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:  # 裁剪
        pp = Crop(0.7, 0.7)
    return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    '''
    x: strong feature
    y: weak feature'''
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

# targeted at boundary: only p/n coexsits.
# learn features that focus on boundary prediction
# use feature vectors to guide pixel prediction 
# covariance to encourage the feature difference in the most decisive ones
def feature_loss(feature_map, pred, kr=4, norm=False, crtl_loss=True, w_ftp=0, topk=16, step_ratio=2):
    '''
    pred: n, 1, h, w'''
    # normalize feature map (but how?)
    if norm:
        fmap = feature_map / feature_map.std(dim=(-1,-2), keepdim=True).mean(dim=1, keepdim=True)
    else: fmap=feature_map  # 16,64,160,160
    # print(fmap.max(), fmap.min(), fmap.std(dim=(-1,-2)).max())

    n, c, h, w =fmap.shape  # n:16, c:64, h:160, w:160
    # get local feature map
    ks = 2*kr  # 8
    assert h%ks==0 and w%ks==0
    # print('ks', ks)
    uf = lambda x: F.unfold(x, ks, padding = 0, stride=ks//step_ratio).permute(0,2,1).reshape(-1, x.shape[1], ks*ks) # N * no.blk, 64, 8*8
    fcmap = uf(fmap)  # 24336,64,64
    fcpred = uf(pred) # N', 1, 10*10  24336,1,64
    # # get fg/bg confident coexisting block
    cfd_thres = .8
    exst = lambda x: (x>cfd_thres).sum(2, keepdim=True) > 0.3*ks*ks
    coexists = (exst(fcpred) & exst(1-fcpred))  # 24336,1,1
    coexists = coexists[:, 0, 0] # N', 1, 1  # 24336
    fcmap = fcmap[coexists]
    fcpred = fcpred[coexists]
    # print(fcmap.shape, fcpred.shape)
    if not len(fcmap):
        return 0, 0
    # minus mean 均值归一化
    mfcmap = fcmap - fcmap.mean(2, keepdim=True)
    mfcpred = fcpred - fcpred.mean(2, keepdim=True)
    # get most relevance in confident area bout saliency
    cov = mfcmap.matmul(mfcpred.permute(0, 2, 1)) # N', 64, 1
    sgnf_id = cov.abs().topk(topk, dim=1)[1].expand(-1,-1,ks*ks) # n', topk, 10*10  topk=16
    sg_fcmap = fcmap.gather(dim=1, index=sgnf_id) # n', topk, 10*10
    # different potential calculation
    crf_k = lambda x: (-(x[:, :, None]-x[:, :, :, None])**2 * 0.5).sum(1, keepdim=True).exp() # n', 1, 100, 100
    pred_grvt = lambda x,y: (1-x)*y + x*(1-y) # (x-y).abs() # x*y + (1-x)*(1-y) - x*(1-y) - (1-x)*y
    ft_grvt = lambda x: 1-crf_k(x)
    # position
    xy = torch.stack(torch.meshgrid(torch.arange(ks, device=pred.device), torch.arange(ks, device=pred.device))) / 6  # 2,8,8
    xy = (xy).reshape(1,2, ks*ks).expand(len(sg_fcmap),-1,-1) # 1, 1, 100  5,2,64
    ffxy = crf_k(xy)  # 5,1,64,64
    if crtl_loss:
        # train the feature map without pred grad
        # L2 norm loss
        pmap = fcpred.detach()
        pmap = 0.5 - pred_grvt(pmap.unsqueeze(2), pmap.unsqueeze(-1)) # n', 1, 100, 100
        fpmap = ft_grvt(sg_fcmap) * ffxy
        ice = (pmap*fpmap).mean()
        # reversely, train the pred map
        # calculate CRF with confident point
        fffm = crf_k(sg_fcmap.detach())
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
    else:
        ice = 0
        fffm = crf_k(sg_fcmap)
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
        kernel[torch.eye(ks*ks, device=pred.device, dtype=bool).expand_as(kernel)] = 0

    pp = pred_grvt(fcpred[:,:,None], fcpred.unsqueeze(-1)) # n', 1, 100, 100  5,1,64,64
    if w_ftp==0:
        crf = (kernel * pp).mean()
    elif w_ftp==1:  # 1
        crf = (kernel.detach() * pp).mean() * (1+w_ftp)
    else:
        crf = unsymmetric_grad(kernel, pp, lambda x,y:(x*y).mean(), 1-w_ftp, 1+w_ftp)
    return crf, ice


def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st = 60, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1, sb=1):
    """
    w_ft:0.15
    ft_st:60
    ft_fct:0.5
    ft_head:
    mtrsf_prob:1
    ops:[0,1,2]
    w_l2g:0.3
    l_me:0.05
    me_st:20
    me_all:False
    multi_sc:0
    l:0.3
    sl:1
    """
    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']
    ### feature loss
    fm = []
    def hook(m, i, o):
        if not ft_head:
            fm.extend(get_featuremap(m, i[0]))
        else:
            fm.append(net.feature_head[0](i[0]))
    hh = net.head[4].register_forward_hook(hook)

    ######  saliency structure consistency loss  ######
    do_moretrsf = np.random.uniform()<mtrsf_prob  # 以1的概率进行数据增强 True
    if do_moretrsf:
        pre_transform = get_transform(ops)  # 随机增强
        image_tr = pre_transform(image)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
    sc_fct = 0.6 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)  # 将变换后的图像缩小至0.6
    out2, _, out3, out4, out5, out6 = net(image, )
    # out2_org = out2
    hh.remove()
    out2_s, _, out3_s, out4_s, out5_s, out6_s = net(image_scale, )

    ### Calc intra_consisten loss (l2 norm) / entorpy
    loss_intra = []
    if epoch>=me_st:  # me_st=20
        def entrp(t):
            etp = -(F.softmax (t, dim=1) * F.log_softmax (t, dim=1)).sum(dim=1)
            msk = (etp<0.5)
            return (etp*msk).sum() / (msk.sum() or 1)
        me = lambda x: entrp(torch.cat((x*0, x), 1)) # orig: 1-x, x
        if not me_all:
            # Inside-view Consistency Loss
            e = me(out2)
            loss_intra.append(e * get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st))
            loss_intra = loss_intra + [0,0,0,0]
            sw.add_scalar('intra_entropy', e.item(), global_step)
        elif me_all:
            ga = get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st)
            for i in [out2, out3, out4, out5, out6]:
                loss_intra.append(me(i)*ga)
            sw.add_scalar('intra_entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    def out_proc(out2, out3, out4, out5, out6):
        a = [out2, out3, out4, out5, out6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]  # 2通道，背景和目标
        return a

    # Synthesize Boundaries
    # if epoch >= sb:  # sb=28
    #     prediction = out2.clone()
    #     # 判断是否是二值图像
    #     prediction = prediction.sigmoid().data.cpu().numpy()
    #     prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
    #     prediction *= 255
    #
    #     region = region.data.cpu().numpy()
    #
    #     # 将图像二值化
    #     removed_prediction = np.zeros_like(prediction)
    #     removed_prediction[prediction>127] = 255
    #
    #     flag_mask = np.logical_and(removed_prediction == 255, region == 255)
    #     removed_prediction[flag_mask] = 0
    #     removed_prediction = torch.from_numpy(removed_prediction).cuda()
    #
    #     syn_out2, _, syn_out3, syn_out4, syn_out5, syn_out6 = net(syn_image, )
    #
    #     syn_prediction, syn_out3, syn_out4, syn_out5, syn_out6 = out_proc(syn_out2, syn_out3, syn_out4, syn_out5, syn_out6)
    #
    #     syn_image_ = F.interpolate(syn_image, scale_factor=0.5, mode='bilinear', align_corners=True)  # 下采样
    #     syn_sample = {'rgb': syn_image_}
    #     syn_out2_ = F.interpolate(syn_prediction[:, 1:2], scale_factor=0.5, mode='bilinear', align_corners=True)
    #     syn_out3_ = F.interpolate(syn_out3[:, 1:2], scale_factor=0.5, mode='bilinear', align_corners=True)
    #     syn_out4_ = F.interpolate(syn_out4[:, 1:2], scale_factor=0.5, mode='bilinear', align_corners=True)
    #     syn_out5_ = F.interpolate(syn_out5[:, 1:2], scale_factor=0.5, mode='bilinear', align_corners=True)
    #     syn_out6_ = F.interpolate(syn_out6[:, 1:2], scale_factor=0.5, mode='bilinear', align_corners=True)
    #
    #     # Local Saliency Coherence Loss (Context Affinity Loss)
    #     loss_lsc_syn = loss_lsc(syn_out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, syn_sample, syn_image_.shape[2], syn_image_.shape[3])['loss']
    #
    #     # Self-consistent Loss (Cross-View Consistency Loss)
    #     removed_prediction_s = F.interpolate(removed_prediction, scale_factor=0.5, mode='bilinear', align_corners=True)
    #
    #     loss_ssc_syn2 = SelfConsistency(syn_out2_, removed_prediction_s, alpha=0.5)
    #     loss_ssc_syn3 = SelfConsistency(syn_out3_, removed_prediction_s, alpha=0.5)
    #     loss_ssc_syn4 = SelfConsistency(syn_out4_, removed_prediction_s, alpha=0.5)
    #     loss_ssc_syn5 = SelfConsistency(syn_out5_, removed_prediction_s, alpha=0.5)
    #     loss_ssc_syn6 = SelfConsistency(syn_out6_, removed_prediction_s, alpha=0.5)
    #
    #
    #     # PCE
    #     syn_gt = syn_mask.squeeze(1).long()
    #     syn_bg_label = syn_gt.clone()
    #     syn_fg_label = syn_gt.clone()
    #     syn_bg_label[syn_gt != 0] = 255  # gt中不为0的像素设置成255
    #     syn_fg_label[syn_gt == 0] = 255  # gt中为0的像素中设置为255
    #
    #     g_syn = get_current_consistency_weight(epoch - sb, consistency=0.005, consistency_rampup=t_epo - sb)
    #
    #     loss_syn2 = g_syn*(loss_lsc_syn + loss_ssc_syn2 + criterion(syn_prediction, syn_bg_label) + criterion(syn_prediction, syn_fg_label))
    #     loss_syn3 = g_syn*(loss_ssc_syn3 + criterion(syn_out3, syn_bg_label) + criterion(syn_out3,syn_fg_label))
    #     loss_syn4 = g_syn*(loss_ssc_syn4 + criterion(syn_out4, syn_bg_label) + criterion(syn_out4, syn_fg_label))
    #     loss_syn5 = g_syn*(loss_ssc_syn5 + criterion(syn_out5, syn_bg_label) + criterion(syn_out5, syn_fg_label))
    #     loss_syn6 = g_syn*(loss_ssc_syn6 + criterion(syn_out6, syn_bg_label) + criterion(syn_out6, syn_fg_label))
    #
    # else:
    #     loss_syn2 = torch.tensor(0)
    #     loss_syn3 = torch.tensor(0)
    #     loss_syn4 = torch.tensor(0)
    #     loss_syn5 = torch.tensor(0)
    #     loss_syn6 = torch.tensor(0)

    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6)
    out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s)

    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        out2_s = out2_s[:, 1:2]
        # out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    else:
        out2_ss = pre_transform(out2)  # 随机变换
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)  # 0.3倍缩放
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)  # 0.5倍缩放

    # Cross-View Consistency Loss
    loss_ssc = (SaliencyStructureConsistency(out2_s, out2_scale.detach(), 0.85) * (w_l2g + 1) + SaliencyStructureConsistency(out2_s.detach(), out2_scale, 0.85) * (1 - w_l2g)) if sl else 0
    
    ######   label for partial cross-entropy loss  ######
    gt = mask.squeeze(1).long()  # unknown:0 foreground:1 background:2
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255  # gt中不为0的像素设置成255
    fg_label[gt == 0] = 255  # gt中为0的像素中设置为255
 
    ## feature loss
    if epoch>=ft_st:
        wl = get_current_consistency_weight(epoch-ft_st, w_ft, t_epo-ft_st)  # float
        # ft_map, bs = fm
        ft_map =(fm[0])  # 16,64,80,80
        pred_s = out2[:, 1:2].clone()  # 16,1,320,320
        pred_s[:,0][gt!=255] = gt[gt!=255].float()
        pred_s = F.interpolate(pred_s, scale_factor = ft_fct, mode='bilinear', align_corners=False)  # 0.5下采样
        # adjust size
        ft_map = F.interpolate(ft_map, out2.shape[-2:], mode='bilinear', align_corners=False)  # 16,64,320,320
        ft_map = F.interpolate(ft_map, pred_s.shape[-2:], mode='bilinear', align_corners=False)  # 16,64,160,160
        # Semantic Significance Loss
        fl, crtl = feature_loss(ft_map, pred_s, **ft_dct)  # crtl=0
        # print('here', loss_ssc, crtl, fl, wl)
        sw.add_scalar('ft_loss', fl.item() if isinstance(fl, torch.torch.Tensor) else fl, global_step=global_step)
        sw.add_scalar('fthead_loss', crtl.item() if isinstance(crtl, torch.torch.Tensor) else crtl, global_step=global_step)
        loss_ssc = loss_ssc + crtl + fl * wl

    ######   local saliency coherence loss (scale to realize large batchsize)  ######
    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)  # 0.25下采样 16,3,80,80
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)  # 16,1,80,80
    # Context Affinity Loss
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss2_lsc + loss_intra[0] #+ 0.05*loss_syn2  ## dominant loss

    # a = type(loss2)
    # b = type(loss_syn)

    ######  auxiliary losses  ######
    out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc + loss_intra[1] #+ 0.05*loss_syn3
    out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc + loss_intra[2] #+ 0.05*loss_syn4
    out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc + loss_intra[3] #+ 0.05*loss_syn5

    out6_ = F.interpolate(out6[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss6_lsc = loss_lsc(out6_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss6 = criterion(out6, fg_label) + criterion(out6, bg_label) + l * loss6_lsc + loss_intra[4] #+ 0.05*loss_syn6

    return loss2, loss3, loss4, loss5, loss6
