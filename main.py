import sys
from config import Config
from trainers import *
from finetuner import *

if __name__ == "__main__":
	try:
		config_path = sys.argv[1]
	except:
		config_path = 'BDW.yaml'
	config = Config(config_path, 0)
	config.print_config()

	# train
	trainer = Trainer(config)
	trainer.train()

	trainer.load_best_model()
	full_val_set = torch.utils.data.ConcatDataset([trainer.data.val_train_set, trainer.data.val_test_set])
	full_val_loader = torch.utils.data.DataLoader(full_val_set,
	    batch_size=trainer.config['data']['test']['batch_size'], shuffle=True, drop_last=True)
	test_loss, test_acc, test_acc_top5 = trainer.test(-1, full_val_loader, trainer.data.test_loader)
	print(f'Logistic regression test accuracy {test_acc:.2f}%', flush=True)
	trainer.save_results(test_loss, test_acc, test_acc_top5)

	# finetune
	finetuner = FineTuner(trainer.config, trainer.data)
	finetuner.run_episodes()

	finetuner.retune()
	test_loss, test_acc, test_acc_top5 = finetuner.test_classifier(-1, finetuner.data.test_loader)
	print(f'Finetuned test accuracy {test_acc:.2f}%', flush=True)
	finetuner.save_results(test_loss, test_acc, test_acc_top5)
