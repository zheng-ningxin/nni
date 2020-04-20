# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
sa_tuner.py including:
    class SATuner
"""

import copy
import logging
import numpy as np

import nni
from nni.tuner import Tuner
from nni.utils import convert_dict2tuple
import nni.parameter_expressions as parameter_expressions


TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'

logger = logging.getLogger('SA_AutoML')


class SATuner(Tuner):
    """
    GridSearchTuner will search all the possible configures that the user define in the searchSpace.
    The only acceptable types of search space are ``choice``, ``quniform``, ``randint``

    Type ``choice`` will select one of the options. Note that it can also be nested.

    Type ``quniform`` will receive three values [``low``, ``high``, ``q``],
    where [``low``, ``high``] specifies a range and ``q`` specifies the interval.
    It will be sampled in a way that the first sampled value is ``low``,
    and each of the following values is 'interval' larger than the value in front of it.

    Type ``randint`` gives all possible intergers in range[``low``, ``high``). Note that ``high`` is not included.
    """

    def __init__(self, optimize_mode='maximize', T=100, stop_temperature=20, cool_down_rate=0.9,
                 random_state=None):
        self.count = -1
        self.space = None
        self.random_state = random_state
        self.state = []
        self.random_init_state()

        self.T = T
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate  # cool down rate

    @property
    def dimension():
        return len(self.space)

    def random_init_state():
        for i, key in enumerate(self.space):
            # TODO:
            self.state.append(parameter_expressions.uniform(
                self.space[key]["_value"][0], self.space[key]["_value"][1], self.random_state))
        self.state = np.array(self.state)

    def _json2parameter(self, ss_spec):
        """
        Generate all possible configs for hyperparameters from hyperparameter space.

        Parameters
        ----------
        ss_spec : dict or list
            Hyperparameter space or the ``_value`` of a hyperparameter

        Returns
        -------
        list or dict
            All the candidate choices of hyperparameters. for a hyperparameter, chosen_params
            is a list. for multiple hyperparameters (e.g., search space), chosen_params is a dict.
        """
        if isinstance(ss_spec, dict):
            if '_type' in ss_spec.keys():
                _type = ss_spec['_type']
                _value = ss_spec['_value']
                chosen_params = list()
                if _type == 'choice':
                    for value in _value:
                        choice = self._json2parameter(value)
                        if isinstance(choice, list):
                            chosen_params.extend(choice)
                        else:
                            chosen_params.append(choice)
                elif _type == 'quniform':
                    chosen_params = self._parse_quniform(_value)
                elif _type == 'randint':
                    chosen_params = self._parse_randint(_value)
                else:
                    raise RuntimeError("Not supported type: %s" % _type)
            else:
                chosen_params = dict()
                for key in ss_spec.keys():
                    chosen_params[key] = self._json2parameter(ss_spec[key])
                return self._expand_parameters(chosen_params)
        elif isinstance(ss_spec, list):
            chosen_params = list()
            for subspec in ss_spec[1:]:
                choice = self._json2parameter(subspec)
                if isinstance(choice, list):
                    chosen_params.extend(choice)
                else:
                    chosen_params.append(choice)
            chosen_params = list(map(lambda v: {ss_spec[0]: v}, chosen_params))
        else:
            chosen_params = copy.deepcopy(ss_spec)
        return chosen_params

    # def _parse_quniform(self, param_value):
    #     """
    #     Parse type of quniform parameter and return a list
    #     """
    #     low, high, q = param_value[0], param_value[1], param_value[2]
    #     return np.clip(np.arange(np.round(low/q), np.round(high/q)+1) * q, low, high)

    # def _parse_randint(self, param_value):
    #     """
    #     Parse type of randint parameter and return a list
    #     """
    #     if param_value[0] >= param_value[1]:
    #         raise ValueError("Randint should contain at least 1 candidate, but [%s, %s) contains none.",
    #                          param_value[0], param_value[1])
    #     return np.arange(param_value[0], param_value[1]).tolist()

    # def _expand_parameters(self, para):
    #     """
    #     Enumerate all possible combinations of all parameters

    #     Parameters
    #     ----------
    #     para : dict
    #         {key1: [v11, v12, ...], key2: [v21, v22, ...], ...}

    #     Returns
    #     -------
    #     dict
    #         {{key1: v11, key2: v21, ...}, {key1: v11, key2: v22, ...}, ...}
    #     """
    #     if len(para) == 1:
    #         for key, values in para.items():
    #             return list(map(lambda v: {key: v}, values))

    #     key = list(para)[0]
    #     values = para.pop(key)
    #     rest_para = self._expand_parameters(para)
    #     ret_para = list()
    #     for val in values:
    #         for config in rest_para:
    #             config[key] = val
    #             ret_para.append(copy.deepcopy(config))
    #     return ret_para

    def update_search_space(self, search_space):
        """
        Check if the search space is valid and expand it: support only ``choice``, ``quniform``, ``randint``.

        Parameters
        ----------
        search_space : dict
            The format could be referred to search space spec (https://nni.readthedocs.io/en/latest/Tutorial/SearchSpaceSpec.html).
        """
        self.space = self._json2parameter(search_space)

    def __array_to_params(self, x):
        """
        array to dict

        maintain int type if the paramters is defined as int in search_space.json
        Parameters
        ----------
        x : numpy array
            array format of parameters

        Returns
        -------
        dict
            dict format of parameters
        """
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(self.dim)
            )

        params = {}
        for i, key in enumerate(self.space):
            params.update({key:  x[i]})

        return params


     def __params_to_array(self, params):
        """
        dict to array

        Parameters
        ----------
        params : dict
            dict format of parameters

        Returns
        -------
        numpy array
            array format of parameters
        """
        try:
            assert set(params.keys()) == set(self.space.keys())
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])


    def generate_parameters(self, parameter_id, **kwargs):
        """
        Generate parameters for one trial.

        Parameters
        ----------
        parameter_id : int
            The id for the generated hyperparameter
        **kwargs
            Not used

        Returns
        -------
        dict
            One configuration from the expanded search space.

        Raises
        ------
        NoMoreTrialError
            If all the configurations has been sent, raise :class:`~nni.NoMoreTrialError`.
        """

        if self.T < self.stop_temperature:
            raise nni.NoMoreTrialError('no more parameters now.')
        # TODO: decrease magnitude
        perturbation = np.random.uniform(-0.1, 0.1, self.dimension)
        self.state = self.state + perturbation
        np.clip(self.state, 0, 1)

        return self.__array_to_params(self.state)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Receive a trial's final performance result reported through :func:`~nni.report_final_result` by the trial.
        GridSearchTuner does not need trial's results.
        """
        if not value:
            return
        if (self.optimize_mode=="maximize" and value > self.current_performance) or (self.optimize_mode=="minimize" and value < self.current_performance):
            self.current_performance = value
            self.state = __params_to_array(parameters)
        else:
            delta_e = np.abs(value - self.current_performance)
            probability = math.exp(-1 * delta_e / self.T)
            if np.random.uniform(0, 1) < probability:
                self.current_performance = value
                self.state = __params_to_array(parameters)
        
        self.T *= self.cool_down_rate

    def import_data(self, data):
        """
        Import additional data for tuning

        Parameters
        ----------
        list
            A list of dictionarys, each of which has at least two keys, ``parameter`` and ``value``
        """
        _completed_num = 0
        for trial_info in data:
            logger.info(
                "Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info(
                    "Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _params_tuple = convert_dict2tuple(_params)
            self.supplement_data[_params_tuple] = True
        logger.info("Successfully import data to grid search tuner.")
