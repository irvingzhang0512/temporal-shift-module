"""
1. 在服务器上运行 tracker：
`python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190`

2. 在Jetbot上连接 tracker：
`python3 -m tvm.exec.rpc_server --tracker=10.0.10.56:9190 --key=1080ti`

3. 运行本程序。得到的部署文件保存在 `logs-{cpu/gpu}-{jetbot/server}-{model-type}-{timestamp}` 中。

"""
import sys
import argparse
import os
import shutil
from datetime import datetime

import tvm
import tvm.relay.testing
from to_relay_model import build_tvm_model
from tvm import autotvm, relay
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner


def _parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model-type", type=str, default="mobilenetv2_online",
                        help="[mobilenetv2_online, resnet50_online]")
    parser.add_argument("--ckpt-path", type=str,
                        default=None)
    parser.add_argument("--num-classes", type=int, default=6)

    # envs
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-remote", action="store_true", default=False)
    parser.add_argument("--use-pytorch-test-model",
                        action="store_true", default=False)

    # auto tuning
    parser.add_argument("--n_trial", type=int, default=2000)
    parser.add_argument("--early_stopping", type=int, default=600)
    parser.add_argument("--tuner-type", type=str, default="xgb",
                        help="[xgb, ga, random, gridsearch]")

    return parser.parse_args()


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(
                               tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def _build_pytorch_model():
    import torch
    import torchvision
    model_name = 'resnet18'
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model = model.eval()

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = 'input0'
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)
    return mod, params


def main(args):
    opj = os.path.join
    sys.path.append("/hdd02/zhangyiyang/temporal-shift-module")

    logs_basename = './logs'

    if args.use_gpu and args.use_remote:
        from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
        set_cuda_target_arch('sm_53')
        target = tvm.target.cuda(model="nano")
        target_host = "llvm -target=aarch64-linux-gnu"
        logs_basename += "-gpu-jetbot-"
    elif args.use_gpu:
        target = tvm.target.cuda()
        target_host = "llvm"
        logs_basename += "-gpu-server-"
    elif args.use_remote:
        target = 'llvm'
        target_host = "llvm -target=aarch64-linux-gnu"
        logs_basename += "-cpu-jetbot-"
    else:
        target = "llvm"
        target_host = "llvm"
        logs_basename += "-cpu-server-"

    dtype = 'float32'
    logs_basename += args.model_type + "-"
    log_dir = logs_basename + str(int(datetime.now().timestamp()*1000))
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    log_file = opj(log_dir, "%s.log" % args.model_type)

    # build tvm relay model
    if args.use_pytorch_test_model:
        mod, params = _build_pytorch_model()
    else:
        mod, params = build_tvm_model(
            args.model_type, args.num_classes, args.ckpt_path)
    print('tvm model built successfully...')

    # extract tasks
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              target_host=target_host,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    if args.use_remote:
        runner = autotvm.RPCRunner(
            'jetbot', '0.0.0.0', 9190,
            number=5, repeat=3,
            timeout=100, min_repeat_ms=150)
    else:
        runner = autotvm.LocalRunner(
            number=20, repeat=3,
            timeout=100, min_repeat_ms=150)
    print("Tuning...")
    tuning_option = {
        'log_filename': log_file,

        'tuner': args.tuner_type,
        'n_trial': args.n_trial,
        'early_stopping': args.early_stopping,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=1000),
            runner=runner),
    }
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, target_host=target_host, params=params)

        lib.export_library(opj(log_dir, 'deploy_lib.tar'))
        with open(opj(log_dir, 'deploy_graph.json'), "w") as fo:
            fo.write(graph)
        with open(opj(log_dir, 'deploy_param.params'), "wb") as fo:
            fo.write(relay.save_param_dict(params))


if __name__ == '__main__':
    main(_parse_args())
