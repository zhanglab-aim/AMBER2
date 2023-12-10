from ... import backend as F
import os
import numpy as np
from ..base import BaseSearcher


def lstm(x, prev_c, prev_h, w):
    ifog = F.matmul(F.concat([x, prev_h], axis=1), w)
    i, f, o, g = F.split(ifog, 4, axis=1)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)
    g = F.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * F.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def proximal_policy_optimization_loss(
    curr_prediction,
    curr_onehot,
    old_prediction,
    old_onehotpred,
    rewards,
    advantage,
    clip_val,
    beta=None,
):
    rewards_ = F.squeeze(rewards, axis=1)
    advantage_ = F.squeeze(advantage, axis=1)

    entropy = 0
    r = 1
    for t, (p, onehot, old_p, old_onehot) in enumerate(
        zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)
    ):
        ll_t = F.log(F.reduce_sum(old_onehot * p))
        ll_0 = F.log(F.reduce_sum(old_onehot * old_p))
        r_t = F.exp(ll_t - ll_0)
        r = r * r_t
        # approx entropy
        entropy += -F.reduce_mean(F.log(F.reduce_sum(onehot * p, axis=1)))

    surr_obj = F.reduce_mean(
        F.abs(1 / (rewards_ + 1e-8))
        * F.minimum(
            r * advantage_,
            F.clip_by_value(r, clip_value_min=1 - clip_val, clip_value_max=1 + clip_val)
            * advantage_,
        )
    )
    if beta:
        # maximize surr_obj for learning and entropy for regularization
        return -surr_obj + beta * (-entropy)
    else:
        return -surr_obj


def get_kl_divergence_n_entropy(
    curr_prediction, curr_onehot, old_prediction, old_onehotpred
):
    """compute approx
    return kl, ent
    """
    kl = []
    ent = []
    for t, (p, onehot, old_p, old_onehot) in enumerate(
        zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)
    ):
        # print(t, old_p, old_onehot, p, onehot)
        kl.append(F.reshape(F.get_metric("kl_div")(old_p, p), [-1]))
        ent.append(
            F.reshape(F.get_loss("binary_crossentropy", y_true=onehot, y_pred=p), [-1])
        )
    return F.reduce_mean(F.concat(kl, axis=0)), F.reduce_mean(F.concat(ent, axis=0))


def parse_action_str(action_onehot, state_space):
    return [
        state_space[i][int(j)]
        for i in range(len(state_space))
        for j in range(len(action_onehot[i][0]))
        if action_onehot[i][0][int(j)] == 1
    ]


def parse_action_str_squeezed(action_onehot, state_space):
    return [state_space[i][action_onehot[i]] for i in range(len(state_space))]


def convert_arc_to_onehot(controller):
    """Convert a categorical architecture sequence to a one-hot encoded architecture sequence

    Parameters
    ----------
    controller : amber.architect.controller
        An instance of controller

    Returns
    -------
    onehot_list : list
        a one-hot encoded architecture sequence
    """
    with_skip_connection = controller.with_skip_connection
    if hasattr(controller, "with_input_blocks"):
        with_input_blocks = controller.with_input_blocks
        num_input_blocks = controller.num_input_blocks
    else:
        with_input_blocks = False
        num_input_blocks = 1
    arc_seq = controller.input_arc
    model_space = controller.model_space
    onehot_list = []
    arc_pointer = 0
    for layer_id in range(len(model_space)):
        onehot_list.append(
            F.squeeze(
                F.one_hot(
                    tensor=arc_seq[arc_pointer], num_classes=len(model_space[layer_id])
                ),
                axis=1,
            )
        )
        if with_input_blocks:
            inp_blocks_idx = (
                arc_pointer + 1,
                arc_pointer + 1 + num_input_blocks * with_input_blocks,
            )
            tmp = []
            for i in range(inp_blocks_idx[0], inp_blocks_idx[1]):
                tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
            onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
        if layer_id > 0 and with_skip_connection:
            skip_con_idx = (
                arc_pointer + 1 + num_input_blocks * with_input_blocks,
                arc_pointer
                + 1
                + num_input_blocks * with_input_blocks
                + layer_id * with_skip_connection,
            )
            tmp = []
            for i in range(skip_con_idx[0], skip_con_idx[1]):
                tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
            onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
        arc_pointer += (
            1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
        )
    return onehot_list


class Buffer:
    """The ordinal buffer class

    Buffer stores the sampled architectures, computed bias-adjusted rewards/advantages, and make feed-dict related through
    get_data method.

    The main attributes of ``Buffer`` can be divided into short-term and long-term: short-term buffers are directly appending
    new entry to the lists; while long-term buffers will concatenate short-term buffers, perform temporal discounts and/or
    adjust for baseline rewards.

    An entire short-term buffer will correspond to one entry in long-term buffer.

    Short-term buffers will be emptied when converted to long-term, and long-term buffers will be kept at a maximum
    length specified by ``max_size`` argument.

    Parameters
    ----------
    max_size : int
        The maximum number of controller steps of sampled architectures to store. Stored data beyond the last ``max_size``
        will be dumped to disk.

    discount_factor : float
        Temporal discount factor. Not used for now.

    ewa_beta : float
        The average is approx. over the past 1/(1-ewa_beta)

    is_squeeze_dim : bool
        If True, controller samples a sequence of tokens, instead of one-hot arrays. Default is False.

    rescale_advantage_by_reward : bool
        If True, the advantage will be divided by reward to rescale it as a proportion of increase/decrease of reward

    clip_advantage : float
        Clip the advantage to the value of ``clip_advantage`` if the advantage exceeds it. Default is 10.0.

    Attributes
    ----------
    r_bias : float
        The moving baseline for reward; used to compute advantage = this_reward - r_bias.

    state_buffer : list
        The list of short-term states; each entry corresponds to one architecture generated by
        ``amber.architect.BaseController.get_action()``.

    action_buffer : list
        The list of short-term categorical-encoded architectures; each entry is a list, corresponding to one architecture generated
        by ``amber.architect.BaseController.get_action()``

    prob_buffer : list
        The list of short-term probabilities; each entry is a list of numpy.ndarray, where each numpy.ndarray is probability.

    reward_buffer : list
        The list of short-term rewards.

    lt_sbuffer : list of numpy.array
        The list of long-term states; each entry is a numpy.array concatenated from several short-term buffer to
        facilitate batching in ``get_data()``.

    lt_abuffer : list of numpy.array
        The list of long-term architectures; each entry is a numpy.array concatenated from several short-term buffer architectures.

    lt_pbuffer : list of list of numpy.array
        The list of long-term probabilities; each entry is a list of numpy.array corresponding to several short-term buffer
        probabilities. Each numpy.array is concatenated probabilities of the same layer and type from several short-term
        entries.

    lt_adbuffer : list of list of floats
        The list of long-term advantages; each entry is floats of rewards corresponding to one short-term buffer.

    lt_rmbuffer : list of floats
        The list of reward mean for each short-term buffer.

    lt_nrbuffer : list of numpy.array
        The list of original rewards for each short-term buffer.

    """

    def __init__(
        self,
        max_size,
        discount_factor=0.0,
        ewa_beta=None,
        is_squeeze_dim=False,
        rescale_advantage_by_reward=False,
        clip_advantage=10.0,
    ):
        self.max_size = max_size
        self.ewa_beta = (
            ewa_beta if ewa_beta is not None else float(1 - 1.0 / self.max_size)
        )
        self.discount_factor = discount_factor
        self.is_squeeze_dim = is_squeeze_dim
        self.rescale_advantage_by_reward = rescale_advantage_by_reward
        self.clip_advantage = clip_advantage
        self.r_bias = None

        # short term buffer storing single trajectory
        self.state_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.reward_buffer = []

        # long_term buffer
        self.lt_sbuffer = []  # state
        self.lt_abuffer = []  # action
        self.lt_pbuffer = []  # prob
        self.lt_adbuffer = []  # advantage
        self.lt_nrbuffer = []  # lt buffer for non discounted reward
        self.lt_rmbuffer = []  # reward mean buffer

    def store(self, state=None, prob=None, action=None, reward=None, *args, **kwargs):
        if state is not None:
            self.state_buffer.append(state)
        if prob is not None:
            self.prob_buffer.append(prob)
        if action is not None:
            self.action_buffer.append(action)
        if reward is not None:
            self.reward_buffer.append(reward)

    def discount_rewards(self):
        """Discount rewards by temporal discount factor. This is not currently used in architecture searching.

        Example
        ----------
        behavior of discounted reward, given ``reward_buffer=[1,2,3]``::

            if buffer.discount_factor=0.1:
                ([array([[1.23]]), array([[2.3]]), array([[3.]])], [1, 2, 3])
            if buffer.discount_factor=0.9:
                ([array([[5.23]]), array([[4.7]]), array([[3.]])], [1, 2, 3])

        """
        discounted_rewards = []
        running_add = 0
        for i, r in enumerate(reversed(self.reward_buffer)):
            # if rewards[t] != 0:   # with this condition, a zero reward_t can get a non-zero
            #    running_add = 0   # discounted reward from its predecessors
            running_add = running_add * self.discount_factor + np.array([r])
            discounted_rewards.append(
                np.reshape(running_add, (running_add.shape[0], 1))
            )

        discounted_rewards.reverse()
        return discounted_rewards, self.reward_buffer

    def finish_path(self):
        """Finish path of short-term buffer and convert to long-term buffer

        An entire short-term buffer will correspond to one entry in long-term buffer. After conversion, this will also
        dump buffers to file and empty the short-term buffer.
        """
        dcreward, reward = self.discount_rewards()
        # advantage = self.get_advantage()

        # get data from buffer
        # TODO: complete remove `state_buffer` as it's useless; ZZ 2020.5.15
        if self.state_buffer:
            state = np.concatenate(self.state_buffer, axis=0)
        old_prob = [np.concatenate(p, axis=0) for p in zip(*self.prob_buffer)]
        if self.is_squeeze_dim:
            action_onehot = np.array(self.action_buffer)
        else:  # action squeezed into sequence
            action_onehot = [
                np.concatenate(onehot, axis=0) for onehot in zip(*self.action_buffer)
            ]
        r = np.concatenate(dcreward, axis=0)
        if not self.lt_rmbuffer:
            self.r_bias = r.mean()
        else:
            self.r_bias = (
                self.r_bias * self.ewa_beta
                + (1.0 - self.ewa_beta) * self.lt_rmbuffer[-1]
            )

        nr = np.array(reward)[:, np.newaxis]

        if self.rescale_advantage_by_reward:
            ad = (r - self.r_bias) / np.abs(self.r_bias)
        else:
            ad = r - self.r_bias
        # if ad.shape[1] > 1:
        #   ad = ad / (ad.std() + 1e-8)

        if self.clip_advantage:
            ad = np.clip(ad, -np.abs(self.clip_advantage), np.abs(self.clip_advantage))

        if self.state_buffer:
            self.lt_sbuffer.append(state)
        self.lt_pbuffer.append(old_prob)
        self.lt_abuffer.append(action_onehot)
        self.lt_adbuffer.append(ad)
        self.lt_nrbuffer.append(nr)

        self.lt_rmbuffer.append(r.mean())

        if len(self.lt_pbuffer) > self.max_size:
            self.lt_sbuffer = self.lt_sbuffer[-self.max_size :]
            self.lt_pbuffer = self.lt_pbuffer[-self.max_size :]
            # self.lt_rbuffer = self.lt_rbuffer[-self.max_size:]
            self.lt_adbuffer = self.lt_adbuffer[-self.max_size :]
            self.lt_abuffer = self.lt_abuffer[-self.max_size :]
            self.lt_nrbuffer = self.lt_nrbuffer[-self.max_size :]
            self.lt_rmbuffer = self.lt_rmbuffer[-self.max_size :]

        self.state_buffer, self.prob_buffer, self.action_buffer, self.reward_buffer = (
            [],
            [],
            [],
            [],
        )

    def get_data(self, bs, shuffle=True):
        """Get a batched data

        size of buffer: (traj, traj_len, data_shape)

        Parameters
        ----------
        bs : int
            batch size

        shuffle : bool
            Randomly shuffle the index before yielding data. Default is True.

        Yields
        -------
        list
            A list of batched data; entries are ordered as: ``[state, prob, arc_seq, advantages, rewards]``. Each entry
            is a batch of data points as specified by ``bs``.
        """

        lt_sbuffer, lt_pbuffer, lt_abuffer, lt_adbuffer, lt_nrbuffer = (
            self.lt_sbuffer,
            self.lt_pbuffer,
            self.lt_abuffer,
            self.lt_adbuffer,
            self.lt_nrbuffer,
        )

        lt_sbuffer = np.concatenate(lt_sbuffer, axis=0)
        lt_pbuffer = [np.concatenate(p, axis=0) for p in zip(*lt_pbuffer)]
        if self.is_squeeze_dim:
            lt_abuffer = np.concatenate(lt_abuffer, axis=0)
        else:
            lt_abuffer = [np.concatenate(a, axis=0) for a in zip(*lt_abuffer)]
        lt_adbuffer = np.concatenate(lt_adbuffer, axis=0)
        lt_nrbuffer = np.concatenate(lt_nrbuffer, axis=0)

        if shuffle:
            slice_ = np.random.choice(
                lt_sbuffer.shape[0], size=lt_sbuffer.shape[0], replace=False
            )
            lt_sbuffer = lt_sbuffer[slice_]
            lt_pbuffer = [p[slice_] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                lt_abuffer = lt_abuffer[slice_]
            else:
                lt_abuffer = [a[slice_] for a in lt_abuffer]
            lt_adbuffer = lt_adbuffer[slice_]
            lt_nrbuffer = lt_nrbuffer[slice_]

        for i in range(0, len(lt_sbuffer), bs):
            b = min(i + bs, len(lt_sbuffer))
            p_batch = [p[i:b, :] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                a_batch = lt_abuffer[i:b]
            else:
                a_batch = [a[i:b, :] for a in lt_abuffer]
            yield lt_sbuffer[i:b, :, :], p_batch, a_batch, lt_adbuffer[
                i:b, :
            ], lt_nrbuffer[i:b, :]


class ResNetArchTokenDecoder:
    def __init__(self, model_space):
        """ResNetArchTokenDecoder is for decoding and encoding neural architectures of neural
        networks with residual connections

        Parameters
        ----------
        model_space : amber.architect.BaseModelSpace
            The model space which architectures are being sampled from
        """
        self.model_space = model_space
        self._num_layers = len(self.model_space)

    def decode(self, arc_seq):
        """Decode a sequence of architecture tokens into operations and res-connections
        """
        start_idx = 0
        operations = []
        res_con = []
        for layer_id in range(self._num_layers):
            operations.append(arc_seq[start_idx])
            if layer_id > 0:
                res_con.append(arc_seq[(start_idx+1) : (start_idx + layer_id + 1)])
            start_idx += layer_id + 1
        return operations, res_con

    def encode(self, operations, res_con):
        """Encode operations and residual connections to a sequence of architecture tokens

        This is the inverse function for `decode`

        Parameters
        ----------
        operations : list
            A list of integers for categorically-encoded operations
        res_con : list
            A list of list where each entry is a binary-encoded residual connections
        """
        operations_ = list(operations)
        arc_seq = [operations_.pop(0)]
        for op, res in zip(operations_, res_con):
            arc_seq.append(op)
            arc_seq.extend(res)
        return arc_seq

    def sample(self, seed=None):
        np.random.seed(seed)
        ops = []
        for _ in range(self._num_layers):
            ops.append(np.random.randint(len(self.model_space[_])))
        res_con = []
        for _ in range(1, self._num_layers):
            res_con.append( np.random.binomial(n=1, p=0.5, size=_).tolist())
        np.random.seed(None)
        return self.encode(operations=ops, res_con=res_con)


class BaseController(BaseSearcher):
    """
    GeneralController for neural architecture search

    This class searches for two steps:
        - computational operations for each layer
        - skip connections for each layer from all previous layers [optional]

    It is a modified version of enas: https://github.com/melodyguan/enas .

    Notable modifications include:
        1) dissection of sampling and training processes to enable better understanding of controller behaviors,
        2) buffering and logging;
        3) loss function can be optimized by either REINFORCE or PPO.


    Parameters
    ----------
    model_space : amber.architect.ModelSpace
        A ModelSpace object constructed to perform architecture search for.

    with_skip_connection : bool
        If false, will not search residual connections and only search for computation operations per layer. Default is
        True.

    share_embedding : dict
        a Dictionary defining which child-net layers will share the softmax and embedding weights during Controller
        training and sampling. For example, ``{1:0, 2:0}`` means layer 1 and 2 will share the embedding with layer 0.

    use_ppo_loss : bool
        If true, use PPO loss for optimization instead of REINFORCE. Default is False.

    kl_threshold : float
        If KL-divergence between the sampling probabilities of updated controller parameters and that of original
        parameters exceeds kl_threshold within a single controller training step, triggers early-stopping to halt the
        controller training. Default is 0.05.

    buffer_size : int
        amber.architect.Buffer stores only the sampled architectures from the last ``buffer_size`` number of from previous
        controller steps, where each step has a number of sampled architectures as specified in ``amber.architect.ControllerTrainEnv``.

    batch_size : int
        How many architectures in a batch to train the controller

    session : F.Session
        The session where the controller tensors is placed

    train_pi_iter : int
        The number of epochs/iterations to train controller policy in one controller step.

    lstm_size : int
        The size of hidden units for stacked LSTM, i.e. controller RNN.

    lstm_num_layers : int
        The number of stacked layers for stacked LSTM, i.e. controller RNN.

    tanh_constant : float
        If not None, the logits for each multivariate classification will be transformed by ``F.tanh`` then multiplied by
        tanh_constant. This can avoid over-confident controllers asserting probability=1 or 0 caused by logit going to +/- inf.
        Default is None.

    temperature : float
        The temperature is a scale factor to logits. Higher temperature will flatten the probabilities among different
        classes, while lower temperature will freeze them. Default is None, i.e. 1.

    optim_algo : str
        Optimizer for controller RNN. Can choose from ["adam", "sgd", "rmsprop"]. Default is "adam".

    skip_target : float
        The expected proportion of skip connections, i.e. the proportion of 1's in the skip/extra
        connections in the output `arc_seq`

    skip_weight : float
        The weight for skip connection kl-divergence from the expected `skip_target`

    name : str
        The name for this Controller instance; all ``F.Tensors`` will be placed under this VariableScope. This name
        determines which tensors will be initialized when a new Controller instance is created.


    Attributes
    ----------
    params : list of F.Variable
        The list of all trainable ``F.Variable`` in this controller

    model_space : amber.architect.ModelSpace
        The model space which the controller will be searching from.

    buffer : amber.architect.Buffer
        The Buffer object stores the history architectures, computes the rewards, and gets feed dict for training.

    session : F.Session
        The reference to the session that hosts this controller instance.

    """

    def __init__(
        self,
        variable_space=None,
        model_space=None,
        with_skip_connection=False,
        share_embedding=None,
        use_ppo_loss=False,
        kl_threshold=0.05,
        buffer_size=15,
        batch_size=5,
        session=None,
        train_pi_iter=20,
        lstm_size=32,
        lstm_num_layers=2,
        tanh_constant=None,
        temperature=None,
        optim_algo="adam",
        skip_target=0.8,
        skip_weight=None,
        rescale_advantage_by_reward=False,
        name="controller",
        verbose=0,
    ):
        super().__init__()
        if variable_space is not None:
            assert model_space is None, ValueError('cannot provide both variable_space and model_space')
            for i in range(len(variable_space)):
                assert variable_space[i].num_choices < np.inf, ValueError(f'invalid variable num_choice at index {i}; please use IntegerModelVariable for controller')
            self.model_space = variable_space
            self.num_layers = len(self.model_space)
            self.num_choices_per_layer = [
                self.model_space[i].num_choices for i in range(self.num_layers)
            ]
        elif model_space is not None:
            self.model_space = model_space
            self.num_layers = len(self.model_space)
            self.num_choices_per_layer = [
                len(self.model_space[i]) for i in range(self.num_layers)
            ]
        else: 
            raise ValueError('must provide at least one of variable space or model space')
        self.share_embedding = share_embedding
        self.with_skip_connection = with_skip_connection

        self.buffer = Buffer(
            max_size=buffer_size,
            # ewa_beta=max(1 - 1. / buffer_size, 0.9),
            discount_factor=0.0,
            is_squeeze_dim=True,
            rescale_advantage_by_reward=rescale_advantage_by_reward,
        )
        self.batch_size = batch_size
        self.verbose = verbose

        self.session = session if session else F.Session()
        self.train_pi_iter = train_pi_iter
        self.use_ppo_loss = use_ppo_loss
        self.kl_threshold = kl_threshold

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight
        if self.skip_weight is not None:
            assert (
                self.with_skip_connection
            ), "If skip_weight is not None, must have with_skip_connection=True"

        self.optim_algo = optim_algo
        self.name = name

        self.create_learner()

    def create_learner(self):
        with F.device_scope("/cpu:0"):
            with F.variable_scope(self.name):
                self._create_weight()
        self.params = [var for var in F.trainable_variables(scope=self.name)]
        self.optimizer = F.get_optimizer(
            self.optim_algo, self.params, opt_config={"lr": 0.001}
        )

    def __str__(self):
        return f"RecurrentRLController(lstm_size={self.lstm_size}, lstm_num_layer={self.lstm_num_layers}, use_ppo_loss={self.use_ppo_loss})"

    def forward(self, input_arc=None):
        is_training = False if input_arc is None else True
        ops_each_layer = 1
        hidden_states = []
        anchors = []
        anchors_w_1 = []
        entropys = []
        probs = []
        log_probs = []
        skip_count = []
        skip_penaltys = []
        if is_training:
            batch_size = F.shape(input_arc[0])[0]
        else:
            batch_size = 1
        prev_c = [
            F.zeros([batch_size, self.lstm_size], F.float32)
            for _ in range(self.lstm_num_layers)
        ]
        prev_h = [
            F.zeros([batch_size, self.lstm_size], F.float32)
            for _ in range(self.lstm_num_layers)
        ]
        skip_targets = F.Variable(
            [1.0 - self.skip_target, self.skip_target],
            trainable=False,
            shape=(2,),
            dtype=F.float32,
        )
        # only expand `g_emb` if necessary
        g_emb_nrow = (
            self.g_emb.shape[0]
            if type(self.g_emb.shape[0]) in (int, type(None))
            else self.g_emb.shape[0].value
        )
        if self.g_emb.shape[0] is not None and g_emb_nrow == 1:
            inputs = F.matmul(F.ones((batch_size, 1)), self.g_emb)
        else:
            inputs = self.g_emb

        arc_pointer = 0
        arc_seq = []

        for layer_id in range(self.num_layers):
            # STEP 1: for each layer, sample operations first by un-rolling RNN
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = F.matmul(
                next_h[-1], self.w_soft["start"][layer_id]
            )  # out_filter x 1
            logit = self._adjust_logits(logit)
            probs.append(F.softmax(logit))

            if is_training:
                token = input_arc[arc_pointer]
            else:
                token = F.multinomial(logit, 1)
            token = F.reshape(token, [batch_size])
            token = F.cast(token, F.int32)
            arc_seq.append(token)

            # sparse NLL/CCE: logits are weights, labels are integers
            log_prob, entropy = self._stepwise_loss(
                tokens=token, logits=logit, batch_size=batch_size
            )
            log_probs.append(log_prob)
            entropys.append(entropy)
            # inputs: get a row slice of [out_filter[i], lstm_size]
            inputs = F.embedding_lookup(self.w_emb["start"][layer_id], token)
            # END STEP 1

            # STEP 2: sample the connections, unless the first layer the number `skip` of each layer grows as layer_id grows
            if self.with_skip_connection:
                if layer_id > 0:
                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

                    query = F.transpose(
                        F.stack(anchors_w_1), [1, 0, 2]
                    )  # batch_size x layer_id x lstm_size
                    # w_attn_2: lstm_size x lstm_size
                    # P(Layer j is an input to layer i) = sigmoid(v^T %*% tanh(W_prev ∗ h_j + W_curr ∗ h_i))
                    query = F.tanh(
                        query
                        + F.expand_dims(F.matmul(next_h[-1], self.w_attn_2), axis=1)
                    )  # query: layer_id x lstm_size
                    query = F.reshape(query, (batch_size * layer_id, self.lstm_size))
                    query = F.matmul(
                        query, self.v_attn
                    )  # query: batch_size*layer_id x 1

                    logit = F.concat([-query, query], axis=1)  # logit: layer_id x 2
                    logit = self._adjust_logits(logit)
                    probs.append(
                        F.reshape(F.softmax(logit), [batch_size, layer_id, 2])
                    )

                    if is_training:
                        skip = input_arc[
                            (arc_pointer + ops_each_layer) : (
                                arc_pointer + ops_each_layer + layer_id
                            )
                        ]
                    else:
                        skip = F.multinomial(logit, 1)  # layer_id x 1 of booleans
                    skip = F.reshape(F.transpose(skip), [batch_size * layer_id])
                    skip = F.cast(skip, F.int32)
                    arc_seq.append(skip)

                    skip_prob = F.sigmoid(logit)
                    kl = skip_prob * F.log(
                        skip_prob / skip_targets
                    )  # (batch_size*layer_id, 2)
                    kl = F.reduce_sum(kl, axis=1)  # (batch_size*layer_id,)
                    kl = F.reshape(kl, [batch_size, -1])  # (batch_size, layer_id)
                    skip_penaltys.append(kl)

                    log_prob, entropy = self._stepwise_loss(
                        tokens=skip, logits=logit, batch_size=batch_size
                    )
                    log_probs.append(log_prob)
                    entropys.append(entropy)

                    skip = F.cast(skip, F.float32)
                    skip = F.reshape(skip, [batch_size, 1, layer_id])
                    skip_count.append(F.reduce_sum(skip, axis=2))

                    # update input as the average of skip-connected prev layer embeddings
                    anchors_ = F.stack(anchors)
                    anchors_ = F.transpose(
                        anchors_, [1, 0, 2]
                    )  # batch_size, layer_id, lstm_size
                    inputs = F.matmul(skip, anchors_)  # batch_size, 1, lstm_size
                    inputs = F.squeeze(inputs, axis=1)
                    # XXX: should add this or not?? what happens for zero skip-connection, inputs is zero?
                    inputs += F.embedding_lookup(self.w_emb["start"][layer_id], token) 
                    inputs /= 1.0 + F.reduce_sum(skip, axis=2)  # batch_size, lstm_size


                anchors.append(next_h[-1])
                # next_h: 1 x lstm_size
                # anchors_w_1: 1 x lstm_size
                anchors_w_1.append(F.matmul(next_h[-1], self.w_attn_1))

            arc_pointer += ops_each_layer + layer_id * self.with_skip_connection
            # END STEP 2
        
        return (
            arc_seq,
            probs,
            log_probs,
            hidden_states,
            entropys,
            skip_count,
            skip_penaltys,
        )

    def sample(self):
        """Get a sampled architecture/action and its corresponding probabilities give current controller policy parameters.

        The generated architecture is the out-going information from controller to manager. which in turn will feedback
        the reward signal for storage and training by the controller.

        Returns
        ----------
        onehots : list
            The sampled architecture sequence. In particular, the architecture sequence is ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]
        probs : list of ndarray
            The probabilities associated with each sampled operation and residual connection. Shapes will vary depending
            on each layer's specification in ModelSpace for operation, and the layer number for residual connections.
        """

        # run forward pass
        ret = self.forward(input_arc=None)
        onehots = F.to_numpy(F.reshape(F.concat(ret[0], axis=0), [-1]))
        probs = [F.to_numpy(x) for x in ret[1]]
        return onehots, probs

    def evaluate(self, input_arc):
        if type(input_arc) is not F.TensorType:
            input_arc = F.Variable(input_arc, trainable=False)
        (
            arc_seq,
            probs,
            log_probs,
            hidden_states,
            entropys,
            skip_count,
            skip_penaltys,
        ) = self.forward(input_arc=input_arc)
        log_probs = F.stack(log_probs)
        onehot_log_prob = F.reshape(
            F.reduce_sum(log_probs, axis=0), [-1]
        )  # (batch_size,)
        if self.with_skip_connection:
            skip_count = F.stack(skip_count)
            skip_penaltys_flat = [
                F.reduce_mean(x, axis=1) for x in skip_penaltys
            ]  # from (num_layer-1, batch_size, layer_id) to (num_layer-1, batch_size); layer_id makes each tensor of varying lengths in the list
            onehot_skip_penaltys = F.reduce_mean(skip_penaltys_flat, axis=0)
        else:
            skip_count = None
            onehot_skip_penaltys = None
        return onehot_log_prob, probs, skip_count, onehot_skip_penaltys

    def train_step(self, input_arc, advantage, old_probs):
        advantage = F.cast(advantage, F.float32)
        old_probs = [F.cast(p, F.float32) for p in old_probs]
        onehot_log_prob, probs, skip_count, onehot_skip_penaltys = self.evaluate(
            input_arc=input_arc
        )
        loss = 0
        if self.with_skip_connection is True and self.skip_weight is not None:
            loss += self.skip_weight * F.reduce_mean(onehot_skip_penaltys)
        if self.use_ppo_loss:
            raise NotImplementedError(f"No PPO support for {F.mod_name} yet")
        else:
            loss += F.reshape(F.tensordot(onehot_log_prob, advantage, axes=1), [])

        self.input_arc = input_arc
        input_arc_onehot = convert_arc_to_onehot(self)
        kl_div, ent = get_kl_divergence_n_entropy(
            curr_prediction=probs,
            old_prediction=old_probs,
            curr_onehot=input_arc_onehot,
            old_onehotpred=input_arc_onehot,
        )
        F.get_train_op(loss=loss, variables=self.params, optimizer=self.optimizer)
        return loss, kl_div, ent

    def train(self):
        """Train the controller policy parameters for one step.

        Returns
        -------
        aloss : float
            Average controller loss for this train step
        """
        self.buffer.finish_path()
        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):
            t = 0
            kl_sum = 0
            ent_sum = 0
            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(
                self.batch_size
            ):
                input_arc = a_batch.T
                curr_loss, curr_kl, curr_ent = self.train_step(
                    input_arc=input_arc, advantage=ad_batch, old_probs=p_batch
                )
                aloss += curr_loss
                kl_sum += curr_kl
                ent_sum += curr_ent
                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold and epoch > 0 and self.verbose > 0:
                    print(
                        "Early stopping at step {} as KL(old || new) = ".format(g_t),
                        kl_sum / t,
                    )
                    return aloss / g_t

            if epoch % max(1, (self.train_pi_iter // 5)) == 0 and self.verbose > 0:
                print(
                    "Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}".format(
                        epoch, aloss / g_t, kl_sum / t, ent_sum / t
                    )
                )

        return aloss / g_t

    def store(self, state=None, prob=None, action=None, reward=None):
        """Store all necessary information and rewards for a given architecture

        This is the receiving method for controller to interact with manager by storing the rewards for a given architecture.
        The architecture and its probabilities can be generated by ``get_action()`` method.

        Parameters
        ----------
        state : list
            The state for which the action and probabilities are drawn.

        prob : list of ndarray
            A list of probabilities for each operation and skip connections.

        action : list
            A list of architecture tokens ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]

        reward : float
            Reward for this architecture, as evaluated by ``amber.architect.manager``

        Returns
        -------
        None

        """
        state = [[[0]]] if state is None else state
        self.buffer.store(
            state=state, prob=prob, action=action, reward=reward
        )

    # private methods below
    def _create_weight(self):
        """Private method for creating tensors; called at initialization"""
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
            with F.variable_scope("lstm/lstm_layer_{}".format(layer_id)):
                w = F.create_parameter(
                    "w", [2 * self.lstm_size, 4 * self.lstm_size], initializer="uniform"
                )
                self.w_lstm.append(w)
        # g_emb: initial controller hidden state tensor; to be learned
        self.g_emb = F.create_parameter(
            "g_emb", [1, self.lstm_size], initializer="uniform"
        )
        # w_emb: embedding for computational operations
        self.w_emb = {"start": []}
        for layer_id in range(self.num_layers):
            with F.variable_scope("lstm/emb/layer_{}".format(layer_id)):
                if self.share_embedding:
                    if layer_id not in self.share_embedding:
                        self.w_emb["start"].append(
                            F.create_parameter(
                                "w_start",
                                [self.num_choices_per_layer[layer_id], self.lstm_size],
                                initializer="uniform",
                            )
                        )
                    else:
                        shared_id = self.share_embedding[layer_id]
                        assert shared_id < layer_id, (
                            "You turned on `share_embedding`, but specified the layer %i "
                            "to be shared with layer %i, which is not built yet"
                            % (layer_id, shared_id)
                        )
                        self.w_emb["start"].append(self.w_emb["start"][shared_id])

                else:
                    self.w_emb["start"].append(
                        F.create_parameter(
                            "w_start",
                            [self.num_choices_per_layer[layer_id], self.lstm_size],
                            initializer="uniform",
                        )
                    )
        # w_soft: dictionary of tensors for transforming RNN hiddenstates to softmax classifier
        self.w_soft = {"start": []}
        for layer_id in range(self.num_layers):
            if self.share_embedding:
                if layer_id not in self.share_embedding:
                    with F.variable_scope("lstm/softmax/layer_{}".format(layer_id)):
                        self.w_soft["start"].append(
                            F.create_parameter(
                                name="w_start",
                                shape=[
                                    self.lstm_size,
                                    self.num_choices_per_layer[layer_id],
                                ],
                                initializer="uniform",
                            )
                        )
                else:
                    shared_id = self.share_embedding[layer_id]
                    assert shared_id < layer_id, (
                        "You turned on `share_embedding`, but specified the layer %i "
                        "to be shared with layer %i, which is not built yet"
                        % (layer_id, shared_id)
                    )
                    self.w_soft["start"].append(self.w_soft["start"][shared_id])
            else:
                with F.variable_scope("lstm/softmax/layer_{}".format(layer_id)):
                    self.w_soft["start"].append(
                        F.create_parameter(
                            "w_start",
                            [self.lstm_size, self.num_choices_per_layer[layer_id]],
                            initializer="uniform",
                        )
                    )
        #  w_attn_1/2, v_attn: for sampling skip connections
        if self.with_skip_connection:
            with F.variable_scope("lstm/attention"):
                self.w_attn_1 = F.create_parameter(
                    "w_1", [self.lstm_size, self.lstm_size], initializer="uniform"
                )
                self.w_attn_2 = F.create_parameter(
                    "w_2", [self.lstm_size, self.lstm_size], initializer="uniform"
                )
                self.v_attn = F.create_parameter(
                    "v", [self.lstm_size, 1], initializer="uniform"
                )
        else:
            self.w_attn_1 = None
            self.w_attn_2 = None
            self.v_attn = None

    def _adjust_logits(self, logit):
        """modify logits to be more smooth, if necessary"""
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * F.tanh(logit)
        return logit

    def _stepwise_loss(self, logits, tokens, batch_size=None):
        # sparse NLL/CCE: logits are weights, labels are integers
        log_prob = F.get_loss(
            "NLLLoss_with_logits", y_true=tokens, y_pred=logits, reduction="none"
        )
        log_prob = F.reshape(log_prob, [batch_size, -1])
        log_prob = F.reduce_sum(log_prob, axis=-1)
        entropy = F.stop_gradient(F.reduce_sum(log_prob * F.exp(-log_prob)))
        return log_prob, entropy
