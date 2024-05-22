from config.configurator import configs
from trainer.trainer import init_seed
from model.build_model import build_model
from trainer.logger import Logger
from utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer

def main():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: training
    best_model = trainer.train(model)

    # Sixth Step: test
    trainer.test(best_model)

def test():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: load model from pretrain_path
    best_model = trainer.load(model)

    # Sixth Step: test
    trainer.test(best_model)

main()


