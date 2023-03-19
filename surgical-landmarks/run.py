import argparse
from dataclasses import asdict
import glob
import json
import os
import torch
from src.common import init_features_path, init_gt, init_model, init_seed, init_vid_list_files, load_json
from src.configs.base import BatchGeneratorConfig, ModelConfig, TrainConfig
from src.dataset.batch_generator import BatchGenerator
import time
from src.pipeline.trainer import Trainer
from src.scripts.visualize_gestures import process_video
import wandb
wandb.init(project="surgical-handmarks")

### Seed Initialization
init_seed()

### Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="apas_tcn")
parser.add_argument('--split', default='1')
parser.add_argument('--gpu', default='0')

parser.add_argument('--features_dim', default='272', type=int)
parser.add_argument('--bz', default='1', type=int)

# 1 = 30 hz, 2 = 15 hz, 3 = 10hz, 5 = 6hz
parser.add_argument('--sample-rate', default='1', type=int)

parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--smoothing-loss-param', default='0.15', type=float)
parser.add_argument('--dropout', default='0.5', type=float)



parser.add_argument('--custom-features', default=None, type=str)
parser.add_argument('--multitask', action='store_true')
parser.add_argument('--task-left-only', action='store_true')
parser.add_argument('--task-right-only', action='store_true')
parser.add_argument('--num_f_maps', default='64', type=int)
parser.add_argument('--num_layers_PG', default='11', type=int)
parser.add_argument('--num_layers_R', default='10', type=int)
parser.add_argument('--num_R', default='3', type=int)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--one-sample', action='store_true')
parser.add_argument('--eval-rate', default=10, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--class-weights', default=None, type=str)
parser.add_argument('--appended-features', default=None, type=str)
parser.add_argument('--excluded-participants', default=None, type=str)

parser.add_argument('--weight-decay', default=None, type=str)

parser.add_argument('--task-custom', default=None, type=str)

parser.add_argument('--mapping-file', default=None, type=str)

parser.add_argument('--benchmark-durations', action='store_true')

parser.add_argument('--append_split_to_features', action='store_true')

# Need input
parser.add_argument('--vid-name', default=None, type=str)
parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--action', type=str, required=True)



args = parser.parse_args()
print(args)


def run(split=args.split, custom_features=args.custom_features):

    # init out

    version = str(time.time())
    output_root = f"./output/{args.dataset}_split{split}_{args.model}_{args.name}_{version}/"

    print(f"saving output to {output_root}")

    model_dir = output_root+"/models/"+args.dataset+"/split_"+split
    results_dir = output_root+"/results/"+args.dataset+"/split_"+split
    log_dir = f"{output_root}/logs.txt"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # end of




    # Init Batch Generator Config

    gt_path, actions_dict, gt_path_tools_left, gt_path_tools_right, mapping_file2, actions_dict2, num_classes = \
        init_gt(args.dataset, args.multitask, args.task_left_only, args.task_right_only, args.task_custom, args.mapping_file)
    sample_rate = args.sample_rate

    features_path = init_features_path(args.dataset, dict(custom_features = custom_features))

    appended_features = load_json(args.appended_features) if args.appended_features else None

    excluded_participants = load_json(args.excluded_participants) if args.excluded_participants else None

    batch_gen_cfg_dict = dict(
        num_classes=num_classes,
        actions_dict=actions_dict,
        gt_path=gt_path,
        features_path=features_path,
        sample_rate=sample_rate, # adjusted in model
        gt_path_tools_left=gt_path_tools_left,
        gt_path_tools_right=gt_path_tools_right,
        actions_dict2=actions_dict2,
        appended_features=appended_features,
        excluded_participants = excluded_participants,
        split = split
    )

    # test_batch_gen_cfg_dict = dict(batch_gen_cfg_dict)
    # test_batch_gen_cfg_dict["sample_rate"] = 1
    # End Batch Generator Config


    # init train config
    gpu = args.gpu
    name = args.name

    num_f_maps = args.num_f_maps
    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    features_dim = args.features_dim

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    
    class_weights = load_json(args.class_weights) if args.class_weights else None
    
    train_cfg_dict = dict(
        device=device,
        actions_dict=actions_dict,
        dataset=args.dataset,
        features_path=features_path,
        gt_path_tools_left=gt_path_tools_left,
        gt_path_tools_right=gt_path_tools_right,
        actions_dict2=actions_dict2,
        log_dir=log_dir,
        split=split,
        lr=args.lr,
        num_epochs=args.num_epochs,
        bz=args.bz,
        gt_path=gt_path,
        model_dir=model_dir,
        resume=args.resume,
        eval_rate=args.eval_rate,
        smoothing_loss_param=args.smoothing_loss_param,
        num_classes=num_classes,
        model=None, # initialized with init_model,
        class_weights=class_weights,
        weight_decay = float(args.weight_decay) if args.weight_decay else None,
        benchmark_durations = args.benchmark_durations or False
    )
    # end of


    # Load model

    model_config = ModelConfig(model=args.model, features_dim=features_dim, num_classes=num_classes, num_f_maps=num_f_maps,
                                    num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                                    actions_dict=actions_dict, actions_dict2=actions_dict2, sample_rate=sample_rate,
                                    dataset=args.dataset,
                                    ckpt=args.ckpt, dropout=args.dropout)
    
    model = init_model(model_config,
                        train_cfg_dict,
                        batch_gen_cfg_dict)

    # wandb.watch(model, log_freq=5, log='all')
    # End Load Model


    # init vid list files

    vid_list_files = init_vid_list_files(dataset=args.dataset, split=split)
    vid_list_file_train = vid_list_files["train"]
    vid_list_file_val = vid_list_files["val"] 
    vid_list_file_tst = vid_list_files["test"] 

    # end of


    train_cfg = TrainConfig(**train_cfg_dict)
    batch_gen_cfg = BatchGeneratorConfig(**batch_gen_cfg_dict)
    
    if args.split != 'all' or (args.split == 'all' and str(split) == '1'):
        try:
            wandb.config.model_config = asdict(model_config)
            wandb.config.train_cfg = train_cfg_dict
            wandb.config.batch_gen_cfg = batch_gen_cfg_dict
            
        except:
            pass

    ### Train

    trainer = Trainer(train_cfg)

    if args.action == "train":

        batch_gen = BatchGenerator(batch_gen_cfg)
        batch_gen.read_data(vid_list_file_train)

        val_batch_gen = BatchGenerator(batch_gen_cfg)
        val_batch_gen.read_data(vid_list_file_val)
        
        test_batch_gen = BatchGenerator(batch_gen_cfg)
        test_batch_gen.read_data(vid_list_file_tst)
        
        if args.one_sample:
            batch_gen.list_of_examples = sorted(batch_gen.list_of_examples)[:1]
            val_batch_gen.list_of_examples = batch_gen.list_of_examples
        trainer.train(train_cfg, batch_gen, val_batch_gen, test_batch_gen)
    elif args.action == "test":
        ckpt = args.ckpt
        if not ckpt:
            raise "--ckpt IS REQUIRED"
        batch_gen = BatchGenerator(batch_gen_cfg)
        batch_gen.read_data(vid_list_file_tst)
        results_g, results_tl, results_tr = trainer.predict2(batch_gen, device=train_cfg.device, sample_rate=batch_gen_cfg.sample_rate, input_data=train_cfg.input_data)
        trainer.evaluate2(results_g, gt_path, verbose=True, wandb_prefix=f"Fixed_Test_split{trainer.current_split}")     
        
    elif args.action == "visualize":
        if not args.ckpt:
            raise "--ckpt IS REQUIRED"
        
        vid_name = args.vid_name
        if not vid_name:
            raise "--vid-name IS REQUIRED"
        
        vids_path = os.getenv("FRONTAL_VIDEOS") #"./data/"+args.dataset+"/smooth_videos/" #"/videos/"
        g2n_path = "./data/"+args.dataset+"/gesture_to_name.json"
        with open(g2n_path) as f:
            gesture_to_name = json.load(f)
        
        batch_gen = BatchGenerator(batch_gen_cfg)
        batch_gen.read_sample(vid_name)
        bt = batch_gen.next_batch(1)
        batch_gen.reset()
        gt_results = bt.get("target")[0]
        results_g, results_tl, results_tr = trainer.predict2(batch_gen, device=train_cfg.device, sample_rate=batch_gen_cfg.sample_rate, input_data=train_cfg.input_data)
        all_vids_in_path = glob.glob(vids_path+"/*")
        vid_path = list(filter(lambda x: x.endswith(".wmv") and vid_name in x and "blank" not in x, all_vids_in_path))[0]
        # vid_path = f"{vids_path}/{vid_name}.wmv.vis.wmv"
        vid_results = results_g[vid_name]
        print(vid_path)
        print(vid_results)
        process_video(vid_path, vid_results, gt_results, "vis_results", gesture_to_name, batch_gen_cfg.actions_dict)

if args.split == "all":
    for i in range(1, 6):
        custom_features = args.custom_features + str(i) if args.append_split_to_features else args.custom_features
        run(split=str(i), custom_features=custom_features)
else:
    run(split=args.split, custom_features=args.custom_features)
