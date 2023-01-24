%%bash
pip install gym[classic_control]==0.25.1
pip install tensorboardX
pip install gym[atari,accept-rom-license]==0.25.1

%load_ext tensorboard
%tensorboard --logdir runs