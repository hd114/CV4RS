#import pandas as pd
#from sklearn.model_selection import train_test_split
#from pathlib import Path
from utils.pytorch_models import ResNet18, ResNet50
#from models.poolformer import create_poolformer_s12
#from models.ConvMixer import create_convmixer
#from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
#from utils.pytorch_utils import start_cuda

import argparse

parser = argparse.ArgumentParser(prog='FLpruning', description='Try fedNISP or fedAvg with random-/f2-pruning')

parser.add_argument('--scenario', help='scenario: 1, 2')
parser.add_argument('--p_strategy', help='pruning strategy: nisp, l2, random, unpruned')
parser.add_argument('--p_round', help='the round in which to start pruning')
parser.add_argument('--p_rate', help='the rate at which to rat 0.01-0.99')
parser.add_argument('--comm', help='number of communication rounds')
parser.add_argument('--epr', help='number of epochs per communication round')



def train():
	args = parser.parse_args()
	print("scenario: 			", args.scenario,
	"\npruning strategy: 	", args.p_strategy,
	"\npruning rate: 		", args.p_rate,
	"\npruning round: 		", args.p_round,
	"\ncommunication rounds:", args.comm,
	"\nepochs per round:	", args.epr)


	scenario = int(args.scenario) # 1 multiple countries per client, 2 one country per client

	pruning_strategy = args.p_strategy # unpruned, random, l2, nisp
	pruning_round = int(args.p_round) #4
	pruning_rate = float(args.p_rate) #0.3
	protected_modules = ["conv1","FC","encoder.4.0.conv1","encoder.4.0.conv2"]#,"encoder.4.0.conv3","encoder.4.0.downsample.0"]

	csv_paths = ["Finland","Ireland","Serbia", "Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"]
	epochs = int(args.epr) #1
	communication_rounds = int(args.comm) #40
	channels = 10
	num_classes = 19
	#model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	#model = create_mlp_mixer(channels, num_classes)
	#model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
    #model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)
	#model = ResNet18("ResNet18", channels=channels, num_cls=num_classes, pretrained=False)
	global_client = GlobalClient(
		scenario=scenario,
		model=model,
		lmdb_path="",
		val_path="",
		csv_paths=csv_paths,
	)
	global_model, global_results = global_client.train(pruning_strategy=pruning_strategy, pruning_round=pruning_round, pruning_rate=pruning_rate, protected_modules=protected_modules, communication_rounds=communication_rounds, epochs=epochs)
	print(global_results)


if __name__ == '__main__':
	train()