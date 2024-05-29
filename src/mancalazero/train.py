import torch
import numpy as np
import time
from mancalazero.gamestate import GameState
from mancalazero.selfplay import SelfPlay
from mancalazero.utils import format_for_torch, SignalHandler
from mancalazero.loss import AlphaZeroLoss
from mancalazero.agent import AlphaZeroAgent, AlphaZeroInitial, RandomAgent


def train(
    Game: GameState,
    Network: torch.nn.Module,
    game_kwargs={},
    network_kwargs={},
    load_weights_from=None,
    selfplay_exploration_moves=30,
    selfplay_load_buffer_from=None,
    selfplay_mcts_kwargs={},
    selfplay_search_kwargs={},
    selfplay_buffer_size=860000,
    selfplay_n_producers=None,
    min_buffer_before_train=430000,
    buffer_savefile=None,
    evaluation_mcts_kwargs={},
    evaluation_search_kwargs={},
    evaluation_n_games=100,
    training_steps=None,
    batch_size=128,
    l2_reg_coeff=0.0001,
    learning_rate=0.01,
    momentum=0.9,
    weights_savefile=None
):
    
    # Handle graceful shutdown on Ctrl+C
    signal_handler = SignalHandler()

    start_weights = None
    if load_weights_from is not None:
        print(f'Loading weights from: {load_weights_from}')
        start_weights = torch.load(load_weights_from)

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
    selfplay.run_buffer(start_message=start_weights)

    # Wait for the buffer to fill to a pre-specified level
    while selfplay.filled < min_buffer_before_train and not signal_handler.stop:
        print(f'\nFilling self-play buffer: {selfplay.game_number} games, {selfplay.filled}/{selfplay_buffer_size} moves')
        signal_handler.sleep(60)
        selfplay.consumer()
        if buffer_savefile is not None:
            selfplay.save_buffer(buffer_savefile)


    model = Network(**network_kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = AlphaZeroLoss(l2_reg_coeff)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    evaluator = SelfPlay(
        Game=Game,
        exploration_moves=0,
        game_kwargs=game_kwargs
    )
    initial_agent=AlphaZeroInitial(
        mcts_kwargs=evaluation_mcts_kwargs,
        search_kwargs=evaluation_search_kwargs
    )
    random_agent=RandomAgent(0.25, 3)


    while selfplay.samples_drawn < training_steps and not signal_handler.stop:


        print(f'\n--- Training step {selfplay.samples_drawn} ---')
        
        # Train network
        sample = selfplay.sample_from_buffer(batch_size)
        inpt, z, pi, mask = format_for_torch(*sample)

        inpt = inpt.to(device)
        z = z.to(device)
        pi = pi.to(device)

        #inpt = inpt/48

        model.train()
        p, v = model(inpt)


        # t, f = model(inpt, mask)
        # print(pi[:5])
        # print(t[:5])
        # input()



        loss = loss_fn(z, v, pi, p, model.named_parameters())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss, current = loss.item(), selfplay.samples_drawn * batch_size + len(inpt)
        print(f"loss: {loss:>7f}  [{current:>5d}/{training_steps:>5d}]")

        weights = model.state_dict()

        # t, f = model(inpt)
        # print(pi[:5])
        # print(t[:5])
        # print(mask[:5])
        # print(z, f)
        # # input()

        # if step > 1000:
        #     break



        # t, f = model(inpt[3:5], mask[3:5])
        # print(pi[3:5])
        # print(t)
        # input()



    if evaluation_n_games:

        # Evaluate new agent against
        model.eval()

        current_agent = AlphaZeroAgent(
            model,
            mcts_kwargs=evaluation_mcts_kwargs,
            search_kwargs=evaluation_search_kwargs
        )
        aza_vs_random = evaluator.tournament(
            [current_agent, random_agent],
            n_games=evaluation_n_games,
            stop_signal_handle=signal_handler.stop_signal,
            move_history=True
        )
        aza_vs_azi = evaluator.tournament(
            [current_agent, initial_agent],
            n_games=evaluation_n_games,
            stop_signal_handle=signal_handler.stop_signal,
            move_history=True
        )
        print('AlphaZero vs Random:')
        print(aza_vs_random)
        print('AlphaZero vs AlphaZeroInitial:')
        print(aza_vs_azi)


    selfplay.terminate()
    if buffer_savefile is not None:
        selfplay.consumer()
        selfplay.save_buffer(buffer_savefile)


    if weights_savefile is not None:
        torch.save(weights, weights_savefile)
        print(f'Model saved to {weights_savefile}')
