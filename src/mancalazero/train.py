import torch
import numpy as np
import time
from mancalazero.gamestate import GameState
from mancalazero.selfplay import SelfPlay
from mancalazero.utils import SignalHandler
from mancalazero.loss import AlphaZeroLoss


def train(
    Game: GameState,
    Network: torch.nn.Module,
    game_kwargs={},
    network_kwargs={},
    selfplay_exploration_moves=30,
    selfplay_load_buffer_from=None,
    selfplay_mcts_kwargs={},
    selfplay_search_kwargs={},
    selfplay_buffer_size=860000,
    selfplay_n_producers=None,
    min_buffer_before_train=430000,
    buffer_savefile=None,
    training_steps=None,
    batch_size=128,
    l2_reg_coeff=0.0001,
    learning_rate=0.01,

):
    
    # Handle graceful shutdown on Ctrl+C
    signal_handler = SignalHandler()

    print('Starting self-play buffer')
    selfplay = SelfPlay(
        Game=Game,
        exploration_moves=selfplay_exploration_moves,
        Network=Network,
        load_buffer_from=selfplay_load_buffer_from,
        game_kwargs=game_kwargs,
        network_kwargs=network_kwargs,
        mcts_kwargs=selfplay_mcts_kwargs,
        search_kwargs=selfplay_search_kwargs,
        buffer_size=selfplay_buffer_size,
        n_producers=selfplay_n_producers
    )
    selfplay.run_buffer()

    # Wait for the buffer to fill to a pre-specified level
    while selfplay.filled < min_buffer_before_train and not signal_handler.stop:
        print(f'\nFilling self-play buffer: {selfplay.game_number} games, {selfplay.filled}/{selfplay_buffer_size} moves')
        signal_handler.sleep(120)
        selfplay.consumer()
        if buffer_savefile is not None:
            selfplay.save_buffer(buffer_savefile)


    model = Network(**network_kwargs)
    loss_fn = AlphaZeroLoss(l2_reg_coeff)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    while selfplay.samples_drawn < training_steps and not signal_handler.stop:

        selfplay.sample_from_buffer(batch_size)

        # model.train()
        # p, v = model(obs, mask)
        # loss = loss_fn(z, v, pi, p, mask, model.named_parameters())

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()


    


    # train_model = MancalaNet()
    # loss_fn = AlphaZeroLoss(l2_reg_coeff)
    # 



    # for step, (obs, z, pi, mask) in enumerate(dataloader):

    #     print(f'Step {step}')
    #     print('Batch creation time: ', round(time.time() - s, 3))
    #     s = time.time()

    #     # train_model.check_all_params()


    #     loss = train_step(obs, z, pi, mask)

    #     print('Loss:', loss.item())
    #     print('Network train time: ',  round(time.time() - s, 3))
    #     s = time.time()


    #     # loss, current = loss.item(), step * batch_size + len(obs)
    #     # print(f"loss: {loss:>7f}  [{current:>5d}/{training_steps:>5d}]")

    #     if step >= training_steps:
    #         break


    selfplay.terminate()