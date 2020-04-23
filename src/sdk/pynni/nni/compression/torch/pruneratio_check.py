# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable



class PruneRatio_Checker:
    
    def __init__(self, model, data):
        self.model = model
        self.named_layers = dict(model.named_modules())
        # data to used to build the topology of the network
        # For example, data, label = next(iter(dataloader))
        self.data = data
        self.hooks = []
        # Following variables are related with the graph
        # Save the tensor/variable nodes
        self.tensors = set() # save the unique id
        self.layers= set()
        self.id2obj = {} # save the mapping relation ship 
        self.forward_edge = {} # save the depedency relation grapgh
        # self.backward_edge = {}
        
        # The functions need to be hooked to build up the graph
        # Variable use the same add operation of Tensor (torch.Tensor.__add__)
        self.func_need_hook = [(torch.Tensor, 'view'), (torch.Tensor, '__add__')]
        
        for attr, func in torch.nn.functional.__dict__:
            if hasattr(func, '__call__'):
                # filter out the funtions 
                self.func_need_hook.append(torch.nn.functional, attr)
        self.ori_func = []
        self.visted = set()
        # Init the hook functions
        self.deploy_hooks()
        out = self.model(data)
        # Clear the hook functions
        self.remove_hooks()

    @property
    def hooks_length(self):
        return len(self.hooks)

    def op_decorator(self, func):
        def new_func(*args, **kwargs):
            inputs = []
            for input in args:
                if isinstance(input, Tensor) or isinstance(input, Variable):
                    inputs.append(input)
            out = func(*args, **kwargs)
            # build the graph
            iids = [id(input) for input in inputs]
            oid = id(out)
            self.tensors.update(iids)
            self.tensors.update(oid)
            if oid not in self.id2obj:
                self.id2obj[oid] = out

            for i in range(len(iids)):
                if iids[i] not in self.id2obj:
                    self.id2obj[iids[i]] = inputs[i]
                if iids[i] not in self.forward_edge:
                    self.forward_edge[iids[i]] = [oid]
                elif oid not in self.forward_edge[iids[i]]:
                    self.forward_edge[iids[i]].append(oid)     
            return out
        return new_func

    def get_forward_hook(self):
        def forward_hook(module, inputs, output):
            checker = module.pruneratio_checker
            linputs = list(inputs)
            # Filter the Tensor or Variable inputs out
            linputs = list(filter(lambda x: isinstance(x, Tensor) or isinstance(x, Variable), linputs))
            iids = [id(input) for input in linputs]
            oid = id(output)
            mid = id(module)
            # For the modules that have multiple submodules, for example(Sequential)
            # They will return at here, because the output tensor is already added 
            # by their submodules. Therefore, we only record the lowest level connection.
            if oid in checker.tensors:
                return 
            self.layers.update(mid)
            self.id2obj[mid] = module
            
            checker.tensors.update(oid)
            checker.id2obj[oid] = output
            for i in range(len(iids)):
                if iids[i] not in checker.tensors:
                    checker.tensors.update(iids[i])
                    checker.id2obj[iids[i]] = linputs[i]
                if iids[i] not in checker.forward_edge:
                    checker.forward_edge[iids[i]] = [mid]
                elif mid not in checker.forward_edge[iids[i]]:
                    checker.forward_edge[iids[i]].append(mid)
            # We need to track the input and output tensors from the model perspective
            self.forward_edge[mid] = [oid]
            module.input_tensors = iids
            module.output_tensor = oid
            
        return forward_hook


    def deploy_hooks(self):
        # Put the checker's reference into the modules
        # make the graph building easier
        for submodel in self.model.modules():
            submodel.pruneratio_checker = self
        forward_hook = self.get_forward_hook()
        # deploy the hooks
        for submodel in self.model.modules():
            hook_handle = submodel.register_forward_hook(forward_hook)
            self.hooks.append(hook_handle)
        # Hook the tensor/variable operators
        for mod, attr in self.func_need_hook:
            ori_func = getattr(mod, attr)
            self.ori_func.append(ori_func)
            new_func = self.op_decorator(ori_func)
            setattr(mod, attr, new_func)
            
        
    def remove_hooks(self):
        for submodel in self.model.modules():
            if hasattr(submodel, 'pruneratio_checker'):
                delattr(submodel, 'pruneratio_checker')
        for hook in self.hooks:
            hook.remove()
        # reset to the original function
        for i, (mod, attr) in enumerate(self.func_need_hook):
            setattr(mod, attr, self.ori_func[i])
        

    def traverse(self, curid, channel):
        """
            Traverse the tensors and check if the prune ratio
            is legal for the network architecture.
        """
        if curid in self.visted:
            # Only the tensors can be visited twice, the conv layers
            # won't be access twice in the DFS progress
            # check if the dimmension is ok
            if self.id2obj[curid].prune['channel'] != channel:
                return False
            return True
        self.visted.update(curid)
        outchannel = channel
        if isinstance(self.id2obj[curid], torch.Tensor):
            torch.Tensor.prune={'channel', channel}
        elif isinstance(self.id2obj[curid], torch.nn.Conv2d):
            conv = self.id2obj[curid]
            outchannel = conv.out_channels * conv.prune['ratio']
        OK = True
        for next in self.forward_edge[curid]:
            re = self.traverse(next, outchannel)
            OK = OK and re
        return OK

    def check(self, ratios):
        """
        input:
            ratios: the prune ratios for the layers
            ratios is the dict, in which the keys are 
            the names of the target layer and the values
            are the prune ratio for the corresponding layers
            For example:
            ratios = {'body.conv1': 0.5, 'body.conv2':0.5}
            Note: the name of the layers should looks like 
            the names that model.named_modules() functions 
            returns.
        """
        for name, ratio in ratios:
            layer = self.named_layers[name]
            if isinstance(layer, nn.Conv2d):
                layer.prune = {'ratio' : ratio}
            elif isinstance(layer, nn.Linear):
                # Linear usually prune the input direction
                # because the output are often equals to the 
                # the number of the classes in the classification 
                # scenario
                layer.prune = { 'ratio' : ratio }
        self.visted.clear()
        # N * C * H * W
        is_legal = self.traverse(data, data.size(1))
        # remove the ratio tag of the tensor
        for name, ratio in ratios:
            layer = self.named_layers[name]
            if hasattr(layer, 'prune'):
                delattr(layer, 'prune')
            if hasattr(layer, 'prune'):
                delattr(layer, 'prune')
        return is_legal
            


    