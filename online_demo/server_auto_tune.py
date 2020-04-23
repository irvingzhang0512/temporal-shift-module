import os

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


def get_network():
    import torch
    from mobilenet_v2_tsm import MobileNetV2
    x = torch.rand(1, 3, 224, 224)
    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]
    model = MobileNetV2()
    scripted_model = torch.jit.trace(model, (x, *shift_buffer)).eval()
    shape_list = [("input0", (1, 3, 224, 224))]
    for i, buffer in enumerate(shift_buffer):
        shape_list.append(("input" + str(i + 1), buffer.size()))
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)

    return mod, params


target = tvm.target.cuda()
network = 'test-tsm-mobilenet-v2'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(
            number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}


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


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params = get_network()
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # # compile kernels with history best records
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with relay.build_config(opt_level=3):
    #         graph, lib, params = relay.build_module.build(
    #             mod, target=target, params=params)

    #     # # export library
    #     # tmp = tempdir()
    #     # filename = "net.tar"
    #     # lib.export_library(tmp.relpath(filename))

    #     # # load parameters
    #     # ctx = tvm.context(str(target), 0)
    #     # module = runtime.create(graph, lib, ctx)

    #     # data_tvm=tvm.nd.array(
    #     #     (np.random.uniform(size=input_shape)).astype(dtype))
    #     # module.set_input('data', data_tvm)
    #     # module.set_input(**params)

    #     # inputs = (
    #     #     tvm.nd.empty((1, 3, 224, 224), ctx=ctx),
    #     #     tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
    #     #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
    #     #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
    #     #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    #     #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    #     #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    #     #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
    #     #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
    #     #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
    #     #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
    #     # )
    #     # # evaluate
    #     # print("Evaluate inference time cost...")
    #     # ftimer=module.module.time_evaluator("run", ctx, number=1, repeat=600)
    #     # prof_res=np.array(ftimer().results) * 1000  # convert to millisecond
    #     # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
    #     #       (np.mean(prof_res), np.std(prof_res)))


tune_and_evaluate(tuning_option)
