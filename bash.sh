rm resutls/* -rf
rm data/* -rf
python scripts/seed_knowledge_base.py
python train.py
python train_no_rag.py
python run_experiments.py