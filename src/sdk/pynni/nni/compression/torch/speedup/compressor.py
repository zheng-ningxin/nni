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
from ..utils.shape_dependency import ADD_TYPES, CAT_TYPE, MUL_TYPES
from .sparsity_conflicts import calc_unmask
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
        # self.masks save the mask tensors pruned by the user and the infered
        # masks of the others modules
        self.masks = torch.load(masks_file, str(self.device))
        # self.internal_result save the internal output of the submodules
        self.internal_result = {}

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

    def _prepare_dummy_input(self, node):
        """
        Prepare the dummy_input for the auto mask inference.
        Parameters
        ----------
        node: NodePyGroup
        Returns
        -------
        dummy_input: list
            List of tensors that will be used as input for the target node.

        """
        _logger.debug('Prepare auto mask inference for node: %s',
                      node.unique_name)

        # prepare the inputs and outputs mask for this node,
        # if there is already a mask in self.masks, then use
        # the original mask tensor, else create a new one.
        inputs_name = node.inputs
        # build the dummy_input, in_masks the target node
        dummy_input = []
        debugnames = []
        for _input in inputs_name:
            if _input not in self.internal_result:
                # if the input debug name is not in self.internal_result,
                # then this node isn't a output tensor of any predecessor
                # nodes. This node is a attribute of the submodule, such as
                # weight or bias, etc. We will skip these tensors.
                # If we don't want this specific judgement here, we can merge
                # the `prim::GetAttr` node of the weight/bias tensor into the key
                # node, such as `conv`.
                # This is caused by the `meage_module_node` function in the
                # _graph_utils.py, because it doesn't merge the prim::GetAttr
                # node into the key node. In current version of _graph_utils.py,
                # we will only merge the nodes that have same scope name, however,
                # the scope name of the correponding prim::GetAttr node of `weight` tensor
                # is None.
                continue
            # TODO why detach??
            dummy_input.append(self.internal_result[_input].detach())
            debugnames.append(_input)
            # v_node = self.debugname_to_value[_input]
            # if isinstance(v_node.type(), torch._C.TensorType) and \
            #         'prim::GetAttr' not in v_node.node().kind():
            #     # Filter the value nodes created by the prim::GetAttr, such as
            #     # weight and bias tensor should be skipped

            #     # print(v_node.type().sizes())
            #     # print(v_node)
            #     # print(v_node.node())
            #     shape = tuple(v_node.type().sizes())
            #     # Note: cannot support the value-dependent models
            #     dummy_input.append((torch.rand(shape).to(self.device), _input))
            #     if _input not in self.masks:
            #         # if the input tensor doesn't have masks, then create one
            #         self.masks[_input] = torch.ones(shape).to(self.device)

        return dummy_input, debugnames

    def update_mask(self, node):
        """
        Update the mask for the target node.
        """
        # get the corresponding auto mask inference class according
        # to the config
        if node.op_type not in AutoMaskInferenceType:
            errmsg = 'The auto inference type of %s is not specified! Please specify the \
                auto mask inference type in constants.py!' % str(node.op_type)
            raise Exception(errmsg)
        AutoMaskInferenceClass = AutoMaskInferenceType[node.op_type]

        # this name is consistent with the name returned by named_modules()
        module_name = node.name
        _logger.info('Update mask for %s', module_name)
        unique_name = node.unique_name
        if unique_name in self.auto_inferences:
            # if the auto inference object already in self.auto_inference, then
            # directly update the previous one
            # self.auto_inferences[unique_name].update()
            _logger.info('Update the indirect sparsity for the %s', unique_name)
            self.auto_inferences[unique_name].update_indirect_sparsity()
            return
        # if it is the first visit to this node, then we create a corresponding auto
        # mask inference object for this node
        dummy_input, input_debugname = self._prepare_dummy_input(node)
        # get the input mask from self.masks
        # Note: the input mask of the successor nodes are
        # already created by the predecessor node
        in_masks = [self.masks[debugname] for debugname in input_debugname]
        if node.type == 'func':
            # we cannot get the runable function directly from the jit traced
            # graph, so we translate it back to python function
            func = jit_to_python_function(node)
            # function doesn't have weights
            _auto_infer = AutoMaskInferenceClass(func, dummy_input, in_masks)
        else:
            # node.type == 'module'
            weight_mask = None
            if module_name in self.masks:
                weight_mask = self.masks[module_name]
            _, module = get_module_by_name(self.bound_model, module_name)
            _auto_infer = AutoMaskInferenceClass(
                module, dummy_input, in_masks, weight_mask)
        self.auto_inferences[unique_name] = _auto_infer
        # _auto_infer.update()
        _auto_infer.update_direct_sparsity()
        # also save the input debug names into the auto_infer
        _auto_infer.input_debugname = input_debugname
        # update the mask tensor and the internal output of the submodules
        # after manually unpack the tuple/list of tensors, the number of the outputs
        # of each node should always be one
        assert len(node.outputs) == 1, "The number of the outputs of %s is not 1" % module_name 
        out_debugname = node.outputs[0]
        # update the output mask into self.masks
        self.masks[out_debugname] = _auto_infer.output_mask
        # update the output result into self.internal_result, so that
        # the successor nodes can take these output tensors as inputs.
        self.internal_result[out_debugname] = _auto_infer.output
        # update the parameter mask of the node
        # print(self.masks.keys())
        self.masks[module_name] = _auto_infer.weight_mask

    def _vnode_to_value(self, c_node):
        """
        translate the C Value node into the values/tensors.
        """
        errmsg = "Only support the torch._C.Value type"
        assert isinstance(c_node, torch._C.Value), errmsg
        if isinstance(c_node.type(), torch._C.TensorType):
            shape = tuple(c_node.type().sizes())
            return torch.rand(shape).to(self.device)
        else:
            value = c_node.toIValue()
            # TODO support more kinds of value node
            errmsg = "Doesn't support convert %s to values", str(cnode.type())
            # currently only support the tensors and constant values
            assert value is not None, errmsg
            return value

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
                # also put the graph input tensor into the internal_result
                # TODO if we can find the corresponding relation between the value node
                # and the dummy_inputs, we can use the inputs value in the dummy_input
                value = self._vnode_to_value(self.debugname_to_value[name])
                self.internal_result[name] = value
                # create the mask tensor for the input value
                if isinstance(self.internal_result[name], torch.Tensor):
                    self.masks[name] = torch.ones_like(value)
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
            self.update_mask(curnode)
            successors = self.torch_graph.find_successors(curnode.unique_name)
            for successor in successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[successor])
        # backward mask inference
        for unique_name in out_degree:
            if out_degree[unique_name] == 0:
                visit_queue.put(self.torch_graph.name_to_node[unique_name])
        while not visit_queue.empty():
            curnode = visit_queue.get()
            self.update_mask(curnode)
            predecessors = self.torch_graph.find_predecessors(
                curnode.unique_name)
            for predecessor in predecessors:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[predecessor])

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


    def need_to_unmask(self, node):
        """
        Check if this node has shape/sparsity conflict. If not, then
        return None, if so, return the values that need to be unmasked.

        Parameters
        ----------
        node: NodePyGroup
            The target node to check if need unmask some values.
        Returns
        -------
        unmask: list
            List of the values that need to be unmasked. In the list, each element
            is a tuple which contains the debugName of the tensor and the correponding
            values that need to be unmask in this tensor. For example, [(1, tensor[0, 1])],
            in this example, we need unmask the sencond value of the tensor 1.
        """
        if node.op_type not in ADD_TYPES and node.op_type not in MUL_TYPES \
            and node.op_type != CAT_TYPE:
            # only abobe operators may invovle shape dependencies
            return None
        unique_name = node.unique_name
        auto_infer = self.auto_inferences[unique_name]
        input_masks = auto_infer.in_masks
        output_mask = auto_infer.output_mask
        unmask = calc_unmask(node, input_masks, output_mask)
        return unmask

    def unmask_chain(self, debugname, t_unmask):
        """
        Unmask the values in the tensor specified by debugname.
        This function will also unmask the related dependent values in the
        predecessor nodes/tensors.
        Parameters
        ---------
        debugname: string
            The debugname of the target tensor.
        unmask: torch.Tensor
            This tensor indicates the values that need to be unmasked in the
            target tensor. This tensor only contains 0 and 1, 1-> need to be unmasked, 0
            -> leave it.
        """
        # find corresponding auto inference object
        node = self.torch_graph.output_to_node[debugname]
        unique_name = node.unique_name
        auto_infer = self.auto_inferences[unique_name]
        debugnames, unmasks = auto_infer.unmask(t_unmask)
        for dname, _unmask in zip(debugnames, unmasks):
            self.unmask_chain(dname, _unmask)


    def resolve_conflicts(self):
        """
        Resolve the shape/mask conflict for the model. Some operators may have shape constraints.
        For example, `add`, the add operation need the input tensors have exactly the same shape.
        If the two input tensors of the add opeartor mask difference values/channels, we need to
        unmask some values/channels and padde zeros to make the shapes of the input tensors are the
        same.
        """
        # build the out_degree table for the nodes in the model
        out_degree = {}
        visit_queue = queue.Queue()
        for node in self.torch_graph.nodes_py.nodes_op:
            successors = self.torch_graph.find_successors(node.unique_name)
            out_degree[node.unique_name] = len(successors)
            if out_degree[node.unique_name] == 0:
                # if this node doesn't have any successor nodes
                visit_queue.put(node)
        # backward traverse the model graph and find the operators that have shape
        # dependencies
        while not visit_queue.empty():
            cur_node = visit_queue.get()
            _auto_infer = self.auto_inferences[cur_node.unique_name]
            _logger.debug('Resolve conflict for %s', cur_node.unique_name)
            unmask = self.need_to_unmask(cur_node)
            if unmask is not None:
                for i, tensor in enumerate(unmask):
                    if tensor is not None:
                        # The reason why we use the input_debugname in the _auto_infer
                        # rather the cur_node.inputs is that the cur_node.inputs have
                        # the parameters of the modules (weight, bias), and this is caused by
                        # the merging rules when we build the TorchModuleGraph. In TorchModuleGraph
                        # we merge the node based on its scope name and the 'prim::GetAttr' node of
                        # weight tensor has no scope name.
                        debugname = _auto_infer.input_debugname[i]
                        self.unmask_chain(debugname, tensor)
            predecessors = self.torch_graph.find_predecessors(cur_node.unique_name)
            for predecessor in predecessors:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[predecessor])

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
