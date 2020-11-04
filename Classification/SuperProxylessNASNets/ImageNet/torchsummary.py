import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from collections import OrderedDict
import numpy as np
import os

from models.SuperProxylessNAS.super_proxyless_AutoML import SuperProxylessNASNets_AutoML
from models.SuperProxylessNAS.super_proxyless_AutoML1 import SuperProxylessNASNets_AutoML1
from models.SuperProxylessNAS.super_proxyless_AutoML2 import SuperProxylessNASNets_AutoML2
from models.SuperProxylessNAS.super_proxyless_AutoML3 import SuperProxylessNASNets_AutoML3
from models.SuperProxylessNAS.modules_AutoML3.mix_op import MixedEdge
from models.SuperProxylessNAS.modules_AutoML3.layers import MBInvertedConvLayer
from models.SuperProxylessNAS.layers3 import MaskedConv2d
from models.SuperProxylessNAS.layers3 import MaskedLinear
from models.SuperProxylessNAS.layers3 import ScaleLayer2d

def summary(model, input_size, batch_size=-1, device="cpu"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    params1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    #print("mask conv",mask)
                    out_channels, in_channels, W, H = module.weight.size()  #[out_channels, in_channels, W, H]
                    #print("out_channels, in_channels, W, H conv", out_channels, in_channels, W, H)

                    params1 += mask.sum().numpy()*W*H
                    #params += params1
                    #print("params weight conv", params1)

                    #BN
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask bias",mask)

                    params1 += np.sum(mask>0)*2
                    params += params1
                    #print("params bias", params1)
                    
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    params1 = 0
                    params1 += torch.prod(torch.LongTensor(list(module.weight.size()))).numpy()
                    params += params1
                    #print("params conv", params1)
                    
                elif isinstance(module, nn.Linear):
                    params1 = 0
                    mask = module._mask[:,:].data.cpu()  #[out_features, in_features]
                    #print("mask linear",mask)
                    #out_channels, in_channels = module.weight.size()  #[out_channels, in_channels]
                    #print("out_channels, in_channels linear", out_channels, in_channels)

                    params1 += mask.sum().numpy()
                    params += params1
                    #print("params weight linear", params1)
    
                summary[m_key]["trainable"] = module.weight.requires_grad

            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    params1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask bias",mask)

                    params1 += np.sum(mask>0)
                    params += params1
                    #print("params bias", params1)
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    params1 = 0
                    params1 += torch.prod(torch.LongTensor(list(module.bias.size()))).numpy()
                    params += params1
                    #print("params bias", params1)
                elif isinstance(module, nn.Linear):
                    params1 = 0
                    mask = module._mask[:,:].data.cpu()  #[out_features, in_features]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels, in_channels]
                    #print("mask bias",mask)

                    params1 += np.sum(mask>0)
                    params += params1
                    #print("params bias", params1)

            summary[m_key]["nb_params"] = params

            flops = 0
            if hasattr(module, "weight") and module.weight is not None:
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    flops1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask conv",mask)
                    _, _, output_height, output_width = output.size()
                    #print("output_height, output_width conv", output_height, output_width)
                    out_channels, in_channels, W, H = module.weight.size()  #[out_channels, in_channels, W, H]
                    #print("out_channels, in_channels, W, H conv", out_channels, in_channels, W, H)

                    for i in mask:
                        flops1 += output_height*output_width*i*W*H
                    #flops += flops1
                    #print("flops weight conv", flops1)

                    #BN
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels, in_channels]
                    #print("mask bias",mask)

                    flops1 += np.sum(mask>0)
                    flops += flops1
                    #print("flops bias", flops1)

                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    flops1 = 0
                    _, _, output_height, output_width = output.size()
                    output_channel, input_channel, kernel_height, kernel_width = module.weight.size()
                    flops1 = output_channel * output_height * output_width * input_channel * kernel_height * kernel_width
                    flops += flops1
                    #print("flops conv", flops1)

                elif isinstance(module, nn.Linear):
                    flops1 = 0
                    mask = module._mask[:,:].data.cpu()  #[out_features, in_features]
                    #print("mask linear",mask)
                    #out_channels, in_channels = module.weight.size()  #[out_channels, in_channels]
                    #print("out_channels, in_channels linear", out_channels, in_channels)

                    flops1 += mask.sum().numpy()
                    flops += flops1
                    #print("flops weight linear", flops1)

                summary[m_key]['weight'] = list(module.weight.size())
            else:
                summary[m_key]['weight'] = 'None'

            if hasattr(module, "bias") and module.bias is not None:
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    flops1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels, in_channels]
                    #print("mask bias",mask)

                    flops1 += np.sum(mask>0)
                    flops += flops1
                    #print("flops bias", flops1)
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    flops1 = 0
                    flops1 = module.bias.numel()
                    flops += flops1
                    #print("flops bias", flops1)

                elif isinstance(module, nn.Linear):
                    flops1 = 0
                    mask = module._mask[:,:].data.cpu()  #[out_features, in_features]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask bias",mask)

                    flops1 += np.sum(mask>0)
                    flops += flops1
                    #print("flops bias", flops1)

            summary[m_key]["flops"] = flops

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("---------------------------------------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25}   {:>25} {:>15} {:>15} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Weight", "Param #", "FLOPs #")
    print(line_new)
    print("===========================================================================================================================")
    total_params = 0
    total_flops = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25}  {:>25} {:>15} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            str(summary[layer]["weight"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["flops"]),
        )
        total_params += summary[layer]["nb_params"]
        total_flops += summary[layer]["flops"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("===========================================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Total flops: {0:,}".format(total_flops))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("---------------------------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("---------------------------------------------------------------------------------------------------------------------------")
    # return summary

##############################################################################################################
if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#####################################################################################
#	model_p = SuperProxylessNASNets_AutoML()
#	model_p = torch.load("model_training_arch9").module.cuda()
#	print("SuperProxylessNASNets:", model_p)
#	model = SuperProxylessNASNets_AutoML1()
#	print("SuperProxylessNASNets_AutoML:", model)
#
#	pretrained_dict = model_p.state_dict()
#	#print("pretrained_dict", pretrained_dict)
#	print("pretrained_dict.keys()", pretrained_dict.keys())
#	model_dict = model.state_dict()
#	#print("model_dict", model_dict)
#	print("model_dict.keys()", model_dict.keys())
#
#	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
#	model_dict.update(pretrained_dict)
#	model.load_state_dict(model_dict)
#	#print("model_dict", model_dict)
#	print("model_training:", model)
#	torch.save(model, "model_training_arch9_")

####################################################################################
#	model_p = SuperProxylessNASNets_AutoML1()
#	model_p = torch.load("model_training").module.cpu()
#	print("SuperProxylessNASNets:", model_p)
#	model = SuperProxylessNASNets_AutoML2().cpu()
#	print("SuperProxylessNASNets_AutoML:", model)
#
#	pretrained_dict = model_p.state_dict()
#	#print("pretrained_dict", pretrained_dict)
#	print("pretrained_dict.keys()", pretrained_dict.keys())
#	model_dict = model.state_dict()
#	#print("model_dict", model_dict)
#	print("model_dict.keys()", model_dict.keys())
#
#	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
#	model_dict.update(pretrained_dict)
#	model.load_state_dict(model_dict)
#	#print("model_dict", model_dict)
#	print("model_training:", model)
#	torch.save(model, "model_training_")

####################################################################################
	model_p = SuperProxylessNASNets_AutoML1().cpu()
	model_p = torch.load("model_training").module.cpu()
	print("SuperProxylessNASNets:", model_p)
	model = SuperProxylessNASNets_AutoML3().cpu()
	print("SuperProxylessNASNets_AutoML:", model)

	pretrained_dict = model_p.state_dict()
	#print("pretrained_dict", pretrained_dict)
	print("pretrained_dict.keys()", pretrained_dict.keys())
	model_dict = model.state_dict()
	#print("model_dict", model_dict)
	print("model_dict.keys()", model_dict.keys())

	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	#print("model_dict", model_dict)
	print("model_training:", model)
	torch.save(model, "model_training_")

#####################################################################################
	model = SuperProxylessNASNets_AutoML3().cpu()
	model = torch.load("model_training_").cpu()
	print("model", model)

	# prune model
	for index, module in enumerate(model.modules()):
	    if module.__str__().startswith('ScaleLayer2d') or module.__str__().startswith('ScaleLayer1d'):
	        print("mask", module.mask)
	        if module.__str__().startswith('ScaleLayer2d'):
	            model.set_conv_mask(index-2, np.where(np.array(module.mask) == 0)[0])

	maskList = []
	#maskIndex = 0
	for layer, (name, module) in enumerate(model._modules.items()):
	    if isinstance(module, nn.ModuleList):
	        #print("layer, (name, module)", layer, name, module)
	        for ly, (nm, md) in enumerate(module._modules.items()):
	            #print("ly, (nm, md)", ly, nm, md)
	            for ly1, (nm1, md1) in enumerate(md._modules.items()):
	                #print("ly1, (nm1, md1)", ly, ly1, nm1, md1)
	                if isinstance(md1,  MBInvertedConvLayer):
	                    for ly2, (nm2, md2) in enumerate(md1._modules.items()):
	                        for ly3, (nm3, md3) in enumerate(md2._modules.items()):
	                            if isinstance(md3, ScaleLayer2d):
	                                maskList = md3._mask.cpu().tolist()
                                 
	                if isinstance(md1, MixedEdge):
	                    #print("ly1, (nm1, md1)", ly, ly1, nm1, md1)
	                    mList = []
	                    for ly2, (nm2, md2) in enumerate(md1._modules.items()):
	                        if isinstance(md2, nn.ModuleList):
	                            for ly3, (nm3, md3) in enumerate(md2._modules.items()):
	                                for ly4, (nm4, md4) in enumerate(md3._modules.items()):
	                                    if (ly >= 1) and (ly4 == 0):
	                                        #print("ly, ly4, (nm4, md4)", ly, ly4, nm4, md4)
	                                        for ly5, (nm5, md5) in enumerate(md4._modules.items()):
	                                            if isinstance(md5, MaskedConv2d):
	                                                for i in np.where(np.array(maskList) == 0):
	                                                    md5._mask[:,i,:,:] = 0
	                                                    #print(md5._mask)
	                                    if (ly > 0) and (ly4 == 2):
	                                        #print("ly4, (nm4, md4)", ly4, nm4, md4)
	                                        for ly5, (nm5, md5) in enumerate(md4._modules.items()):
	                                            #print("ly5, (nm5, md5)", ly5, nm5, md5)
	                                            if isinstance(md5, ScaleLayer2d):
	                                                mask = md5._mask.cpu().tolist()
	                                                mList.append(mask)
	                    maskList = [mList[0][i] or mList[1][i] or mList[2][i] or mList[3][i] or mList[4][i] or mList[5][i] for i in range(len(mList[0]))]
	                    #maskIndex = maskIndex + 1
	                    #print("maskIndex, maskList", maskIndex, maskList)
	    elif name == 'feature_mix_layer':
	        #print("layer, (name, module)", layer, name, module)
	        for ly, (nm, md) in enumerate(module._modules.items()):
	            if isinstance(md, MaskedConv2d):
	                for i in np.where(np.array(maskList) == 0):
	                    md._mask[:,i,:,:] = 0
	            if isinstance(md, ScaleLayer2d):
	                maskList = md._mask.cpu().tolist()
	    elif name == 'classifier':
	        #print("layer, (name, module)", layer, name, module)
	        for ly, (nm, md) in enumerate(module._modules.items()):
	            if isinstance(md, MaskedLinear):
	                for i in np.where(np.array(maskList) == 0):
	                    md._mask[:,i] = 0

	torch.save(model, "model_training___")

######################################################################################
	#summary(model, (1, 28, 28), device="cpu")
	#summary(model, (3, 32, 32), device="cpu")
	summary(model, (3, 224, 224), device="cpu")