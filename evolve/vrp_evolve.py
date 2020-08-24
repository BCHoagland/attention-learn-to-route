import argparse
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from pathlib import Path

from utils import load_model
from problems import CVRP


def fitness(dataset, model, params):
    # Need a dataloader to batch instances
    dataloader = DataLoader(dataset, batch_size=1000)

    # Make var works for dicts
    batch = next(iter(dataloader))
    
    # Set the model parameters
    vector_to_parameters(params, model.parameters())

    # Run the model
    # model.eval()      #? keep?
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi = model(batch, return_pi=True)

    return length.mean()


def train(num_epochs, num_samples, vis_iter, save_iter, sigma, lr, problem_size, dataset_size, mirror_sampling, load_source):
    # setup
    torch.manual_seed(1234)

    savedir = f'saved_params/cvrp_{problem_size}'
    Path(savedir).mkdir(parents=True, exist_ok=True)

    model, _ = load_model(f'pretrained/cvrp_{problem_size}/')

    if load_source is not None:
        params = torch.load(f'{savedir}/{load_source}')
        print(f'Starting with pretrained model {load_source}')
    else:
        params = torch.load(f'{savedir}/base')
        print('Starting with untrained model')
    params.requires_grad = False

    dataset = CVRP.make_dataset(size=problem_size, num_samples=dataset_size)

    # report starting fitness
    print(f'Fitness started at \t{fitness(dataset, model, params).item()}')
    print('-' * 19)

    # training loop
    fitness_history = []
    for epoch in range(num_epochs):

        # estimate gradient
        grad = 0
        for _ in range(num_samples):
            eps = torch.randn_like(params)
            grad += fitness(dataset, model, params + sigma * eps) * eps
            if mirror_sampling:
                grad += fitness(dataset, model, params - sigma * eps) * (-eps)

        if mirror_sampling:
            grad /= 2 * num_samples * sigma
        else:
            grad /= num_samples * sigma

        # update parameters by following gradient
        params -= lr * grad

        # print current fitness
        if epoch % vis_iter == vis_iter - 1:
            f = fitness(dataset, model, params).item()
            print(f'Epoch {epoch + 1}/{num_epochs} \t\t{f}')
            fitness_history.append(f)

        # save current parameters
        if epoch % save_iter == save_iter - 1:
            torch.save(params, f'{savedir}/epoch_{epoch + 1}')
            with open(f'{savedir}/fitness_history', 'w') as f:
                f.write(str(fitness_history))


parser = argparse.ArgumentParser(description='Finetune the trained attention model using OpenAI\'s natural evolution strategy')
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--samples', default=10, type=int)
parser.add_argument('--vis_iter', default=1, type=int)
parser.add_argument('--save_iter', default=10, type=int)
parser.add_argument('--sigma', default=0.01, type=float)
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--problem_size', default=100, type=int)
parser.add_argument('--dataset_size', default=100, type=int)
parser.add_argument('--mirror_sampling', default=True, type=bool)
parser.add_argument('--load_source', '-l', default=None, type=str)
args = parser.parse_args()

train(args.epochs, args.samples, args.vis_iter, args.save_iter, args.sigma, args.lr, args.problem_size, args.dataset_size, args.mirror_sampling, args.load_source)
