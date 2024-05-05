from .sacd import SACD
from .dqn import DQN
from .dqn_reimp import DQN_Reimplemented

RL_ALGORITHMS = {
    SACD.name: SACD,
    DQN.name: DQN,
    DQN_Reimplemented.name : DQN_Reimplemented
}
