# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import queue
import logging
import torch
from nni.compression.torch.utils.mask_conflict import fix_mask_conflict
from .compress_modules import replace_module
from .infer_shape import ModuleMasks, infer_from_mask, infer_from_inshape, infer_from_outshape
from .infer_mask import AutoMaskInference
from .jit_translate import jit_to_python_function
from .constants import AutoMaskInferenceType
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module


class ModelSpeedup:
    """
    This class is to speedup the model with provided weight mask
    """

    def __init__(self, model, dummy_input, masks_file):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on the right device before pass in
        masks_file : str
            The path of user provided mask file
        map_location : str
            the device on which masks are placed, same to map_location in ```torch.load```
        """
        from nni._graph_utils import build_module_graph

        self.bound_model = model
        self.inferred_masks = dict()  # key: module_name, value: ModuleMasks
        self.dummy_input = dummy_input
        self.torch_graph = build_module_graph(model, dummy_input)
        # dict object to save the auto inferences objects of the submodules
        self.auto_inferences = {}
        # the index dict to find the corresponding torch._C.Value object
        # according to the debug name
        # we need the dummy_input to infer the mask automaticlly, so we save
        # the indexes from tensor's debugname to the torch._C.Value object.
        self.debugname_to_value = {}
        # device to run the forward inference
        self.device = dummy_input.device
        # load the mask tensor to the same device with the dummy_input
        self.masks = torch.load(masks_file, str(self.device))

    # def infer_module_mask(self, module_name, last_module, mask=None, in_shape=None, out_shape=None):
    #     """
    #     Infer input shape / output shape based on the module's weight mask / input shape / output shape.

    #     For a module:
    #         Infer its input and output shape from its weight mask
    #         Infer its output shape from its input shape
    #         Infer its input shape from its output shape

    #     If its input shape is changed, continue infering its predecessors
    #     If its output shape is changed, continue infering its successors

    #     Parameters
    #     ----------
    #     module_name : str
    #         The name of the node
    #     last_module : str
    #         The name of last visited node
    #     mask : tensor of mask or ModuleMasks
    #         Mask of the weights in this node (i.e., module)
    #     in_shape : ModuleMasks
    #         Input shape of this node
    #     out_shape : ModuleMasks
    #         Output shape of this node
    #     """
    #     input_cmask = output_cmask = None
    #     if module_name in self.inferred_masks:
    #         module_masks = self.inferred_masks[module_name]
    #     else:
    #         module_masks = ModuleMasks(module_name)
    #         self.inferred_masks[module_name] = module_masks

    #     m_type = self.torch_graph.name_to_node[module_name].op_type
    #     _logger.debug("infer mask of module %s with op_type %s", module_name, m_type)
    #     if mask is not None:
    #         _logger.debug("mask is not None")
    #         if not m_type in infer_from_mask:
    #             raise RuntimeError(
    #                 "Has not supported infering input/output shape from mask for module/function: `{}`, {}"
    #                 .format(m_type, module_name))
    #         input_cmask, output_cmask = infer_from_mask[m_type](module_masks, mask)
    #     if in_shape is not None:
    #         _logger.debug("in_shape is not None")
    #         if not m_type in infer_from_inshape:
    #             raise RuntimeError(
    #                 "Has not supported infering output shape from input shape for module/function: `{}`, {}"
    #                 .format(m_type, module_name))
    #         if m_type in ['aten::view', 'aten::flatten', 'aten::mean', 'aten::reshape']:
    #             output_cmask = infer_from_inshape[m_type](module_masks,
    #                                                       in_shape,
    #                                                       self.torch_graph.name_to_node[module_name].auxiliary)
    #         elif m_type in ['aten::cat']:
    #             # To calculate the mask for concat operation, the output shape
    #             # , cat dimension, and the order of the input parameters.
    #             output_cmask = infer_from_inshape[m_type](module_masks,
    #                                                       in_shape,
    #                                                       self.torch_graph.name_to_node[module_name].auxiliary,
    #                                                       last_module)
    #         else:
    #             output_cmask = infer_from_inshape[m_type](module_masks, in_shape)
    #     if out_shape is not None:
    #         _logger.debug("out_shape is not None")
    #         if not m_type in infer_from_outshape:
    #             raise RuntimeError(
    #                 "Has not supported infering input shape from output shape for module/function: `{}`, {}"
    #                 .format(m_type, module_name))
    #         input_cmask = infer_from_outshape[m_type](module_masks, out_shape)

    #     if input_cmask:
    #         predecessors = self.torch_graph.find_predecessors(module_name)
    #         for _module_name in predecessors:
    #             self.infer_module_mask(_module_name, module_name, out_shape=input_cmask)
    #     if output_cmask:
    #         successors = self.torch_graph.find_successors(module_name)
    #         for _module_name in successors:
    #             self.infer_module_mask(_module_name, module_name, in_shape=output_cmask)

    def _prepare_auto_inference(self, node):
        """
        Prepare the dummy_input and the input mask, weight mask, output mask
        for the target node. If the mask tensor is alread created by other nodes,
        then read the mask tensor from self.mask, else create a all-ones mask tensor
        and put it in self.masks.
        Parameters
        ----------
        node: NodePyGroup
        Returns
        -------
        dummy_input: list
            List of tensors that will be used as input for the target node.
        in_mask: list
            List of mask tensors. The length of in_mask is same with dummy_input.
        weight_mask: dict
            Dict object that stores the weight mask of the module, if the node corresponds
            a function, then weight_mask is {}, else the key is the parameter name(such as
            weight, bias), and the value is the mask tensor for the corresponding parameters.
        output_mask: list
            List of mask tensors. The mask tensor of the output tensors, the length of this
            list should be the same with the number of the output tensors.
        """
        _logger.debug('Prepare auto mask inference for node: %s',
                      node.unique_name)
        module_name = node.name
        unique_name = node.unique_name
        # prepare the inputs and outputs mask for this node,
        # if there is already a mask in self.masks, then use
        # the original mask tensor, else create a new one.
        inputs_name = node.inputs

        outputs_name = node.outputs
        # build the dummy_input, in_masks the target node
        dummy_input = []
        in_mask = []
        for _input in inputs_name:
            v_node = self.debugname_to_value[_input]
            if isinstance(v_node.type(), torch._C.TensorType) and \
                'prim::GetAttr' not in v_node.node().kind():
                # This value node should not be created by the prim::GetAttr, such as
                # weight and bias tensor should be skipped

                # print(v_node.type().sizes())
                # print(v_node)
                # print(v_node.node())
                shape = tuple(v_node.type().sizes())
                # Note: cannot support the value-dependent models
                dummy_input.append(torch.rand(shape).to(self.device))
                if _input not in self.masks:
                    self.masks[_input] = torch.ones(shape).to(self.device)
                in_mask.append(self.masks[_input])

        # prepare the parameter mask tensors
        weight_mask = dict()
        if module_name in self.masks:
            weight_mask = self.masks[module_name]

        # prepare the output mask tensors
        output_mask = []
        for _output in outputs_name:
            v_node = self.debugname_to_value[_output]
            if isinstance(v_node.type(), torch._C.TensorType):
                shape = tuple(v_node.type().sizes())
                if _output not in self.masks:
                    _output_mask = torch.ones(shape).to(self.device)
                    self.masks[_output] = _output_mask
                output_mask.append(self.masks[_output])
        return dummy_input, in_mask, weight_mask, output_mask

    def _update_mask(self, node):
        """
        Update the mask for the target node.
        """
        module_name = node.name
        _logger.info('Update mask for %s', module_name)
        unique_name = node.unique_name
        if unique_name in self.auto_inferences:
            # if the auto inference object already in self.auto_inference, then
            # directly use the previous one
            self.auto_inferences[unique_name].update()
            return
        # if it is the first visit to this node, then we create a corresponding auto
        # mask inference object for this node
        dummy_input, in_masks, weight_mask, output_mask = self._prepare_auto_inference(
            node)
        # get the corresponding auto mask inference class according
        # to the config
        if node.op_type not in AutoMaskInferenceType:
            errmsg = 'The auto inference type of %s is not specified! Please specify the \
                auto mask inference type in constants.py!' % str(node.op_type)
            raise Exception(errmsg)
        AutoMaskInferenceClass = AutoMaskInferenceType[node.op_type]
        # this name is consistent with the name returned by named_modules()
        if node.type == 'func':
            # we cannot get the runable function directly from the jit traced
            # graph, so we translate it back to python function
            func = jit_to_python_function(node)
            # function doesn't have weights
            _auto_infer = AutoMaskInferenceClass(
                func, dummy_input, in_masks, weight_mask, output_mask)
        else:
            # node.type == 'module':
            _, module = get_module_by_name(self.bound_model, module_name)
            _auto_infer = AutoMaskInferenceClass(
                module, dummy_input, in_masks, weight_mask, output_mask)
        self.auto_inferences[unique_name] = _auto_infer
        _auto_infer.update()

    def infer_modules_masks(self):
        """
        Infer the mask for all layers in the module, this function can be divided into
        two steps: first, forward inference of the the masks. Second, backward inference
        of the mask. We keep repeating these two steps until the masks of the model doesn't
        change.
        """
        # unpack the tensor tuple/list before the mask inference
        self.torch_graph.unpack_manually()
        # find the input/ouput tensor of the whole graph
        graph_input = []
        graph_output = []
        for name, nodeio in self.torch_graph.nodes_py.nodes_io.items():
            if nodeio.input_or_output == 'input':
                graph_input.append((name, nodeio))
            elif nodeio.input_or_output == 'output':
                graph_output.append((name, nodeio))
        # count the degree for the node in the graph
        in_degree = {}
        out_degree = {}
        visit_queue = queue.Queue()
        for node in self.torch_graph.nodes_py.nodes_op:
            successors = self.torch_graph.find_successors(node.unique_name)
            out_degree[node.unique_name] = len(successors)
            predecessors = self.torch_graph.find_predecessors(node.unique_name)
            in_degree[node.unique_name] = len(predecessors)
            if in_degree[node.unique_name] == 0:
                visit_queue.put(node)
        # Forward mask inference
        while not visit_queue.empty():
            curnode = visit_queue.get()
            # forward mask inference for curnode
            self._update_mask(curnode)
            successors = self.torch_graph.find_successors(curnode.unique_name)
            for successor in successors:
                in_degree[successor.unique_name] -= 1
                if in_degree[successor.unique_name] == 0:
                    visit_queue.put(successor)
        # backward mask inference
        for node in out_degree:
            if out_degree[node] == 0:
                visit_queue.put(node)
        while not visit_queue.empty():
            curnode = visit_queue.get()
            self._update_mask(curnode)
            predecessors = self.torch_graph.find_predecessors(
                curnode.unique_name)
            for predecessor in predecessors:
                out_degree[predecessor.unique_name] -= 1
                if out_degree[predecessor.unique_name] == 0:
                    visit_queue.put(predecessor)

        # Backwards mask inference
        # for module_name, mask in self.masks.items():
        #     _logger.debug('Start mask inference from %s', module_name)
        #     if module_name not in self.torch_graph.name_to_node:
        #         # this module is not traced in the torch_graph,
        #         # jit.trace only correctly records functions and
        #         # modules which are not data dependent (e.g., do
        #         # not have conditionals on data in tensors)
        #         # so, if a node is not traced, we just skip it.
        #         _logger.warning('%s has mask, but not found in the traced graph, just skip it.', module_name)
        #         continue
        #     self.infer_module_mask(module_name, None, mask=mask)

    # def replace_compressed_modules(self):
    #     """
    #     Replace all the modules that have changed (weights/inputs/output) shape.
    #     The new module is created using the same arguments of the to-be-replaced module,
    #     and correctly inherits its weights.

    #     NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
    #     is that ```func``` should be not required to be replaced.
    #     """
    #     for module_name in self.inferred_masks:
    #         g_node = self.torch_graph.name_to_node[module_name]
    #         _logger.debug("replace %s, in %s type, with op_type %s",
    #                       module_name, g_node.type, g_node.op_type)
    #         if g_node.type == 'module':
    #             super_module, leaf_module = get_module_by_name(self.bound_model, g_node.name)
    #             m_type = g_node.op_type
    #             if not m_type in replace_module:
    #                 raise RuntimeError("Has not supported replacing the module: `{}`".format(m_type))
    #             _logger.info("replace module (name: %s, op_type: %s)", g_node.name, m_type)
    #             compressed_module = replace_module[m_type](leaf_module, self.inferred_masks[module_name])
    #             setattr(super_module, g_node.name.split('.')[-1], compressed_module)
    #         elif g_node.type == 'func':
    #             _logger.info("Warning: cannot replace (name: %s, op_type: %s) which is func type",
    #                          module_name, g_node.op_type)
    #         else:
    #             raise RuntimeError("Unsupported node type: {}".format(g_node.type))

    def resolve_conflicts(self):
        """
        resolve the shape/mask conflict.
        """
        pass

    def initialize_speedup(self):
        """
        Do some initial work for speedup.
        """
        # initialize the self.debugname_to_value
        # build a mapping table from the debug name of the tensor
        # to its value node in the graph
        traced_graph = self.torch_graph.trace.graph
        for node in traced_graph.nodes():
            for _input in node.inputs():
                debug_name = _input.debugName()
                if debug_name not in self.debugname_to_value:
                    self.debugname_to_value[debug_name] = _input
            for _output in node.outputs():
                debug_name = _output.debugName()
                if debug_name not in self.debugname_to_value:
                    self.debugname_to_value[debug_name] = _output

    def speedup_model(self):
        """
        There are basically two steps: first, do mask/shape inference,
        second, replace modules.
        """

        _logger.info("start to speed up the model")
        self.initialize_speedup()
        training = self.bound_model.training
        _logger.info("infer module masks...")
        self.infer_modules_masks()
        _logger.info('resolve the mask conflict')
        self.resolve_conflicts()
        _logger.info("replace compressed modules...")
        # self.replace_compressed_modules()
        self.bound_model.train(training)
        _logger.info("speedup done")
