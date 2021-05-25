#!/usr/bin/env python
#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""A sample program highlighting usage of VariantWorks SDK to write a simple SNP variant caller using a CNN."""

import argparse

import nemo
import os
import torch
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import eval_epochs_done_callback

from variantworks.dataloader import HDFDataLoader
from variantworks.neural_types import SummaryPileupNeuralType, HaploidNeuralType

import create_model


class CategoricalAccuracy(object):
    """Categorical accuracy metric."""

    def __init__(self, accumulate=False):
        """Constructor for metric.

        Args:
            accumulate : Accumulate metrics over multiple calls.
        """
        self._num_correct = 0.0
        self._num_examples = 0.0
        self._accumulate = accumulate

    def __call__(self, y_true, y_pred):
        """Compute categorical accuracy."""
        indices = torch.max(y_pred, -1)[1]
        correct = torch.eq(indices, y_true).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]
        acc = self._num_correct / self._num_examples
        if not self._accumulate:
            self._num_correct = 0.0
            self._num_examples = 0.0
        return acc

    def reset(self):
        """Reset counters."""
        self._num_correct = 0.0
        self._num_examples = 0.0


def generate_eval_callback(categorical_accuracy_func):
    """Custom callback function generator for network evaluation loop."""
    def eval_iter_callback(tensors, global_vars):
        if "eval_loss" not in global_vars.keys():
            global_vars["eval_loss"] = []
        for kv, v in tensors.items():
            if kv.startswith("loss"):
                global_vars["eval_loss"].append(torch.mean(torch.stack(v)).item())

        if "top1" not in global_vars.keys():
            global_vars["top1"] = [0.0]
            categorical_accuracy_func.reset()

        output = None
        labels = None
        for kv, v in tensors.items():
            if kv.startswith("output"):
                output = torch.cat(tensors[kv])
            if kv.startswith("label"):
                labels = torch.cat(tensors[kv])

        if output is None:
            raise Exception("output is None")

        with torch.no_grad():
            accuracy = categorical_accuracy_func(labels, output)
            global_vars["top1"][0] = accuracy

    return eval_iter_callback


def train(args):
    """Train a sample model with test data."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU,
        local_rank=args.local_rank)

    model = create_model.create_rnn_model(args.input_feature_size,
                                          args.num_output_logits,
                                          args.gru_size,
                                          args.gru_layers)
    print("model.apply_softmax",model.apply_softmax)

    encoding_dims = ('B', 'W', 'C')
    label_dims = ('B', 'W')
    encoding_neural_type = SummaryPileupNeuralType()
    label_neural_type = HaploidNeuralType()

    # Create train DAG
    my_batch_size = 32
    # numberOfPoints = my_batch_size*step
    print("batch_size",my_batch_size)
    train_dataset = HDFDataLoader(args.train_hdf, batch_size=my_batch_size,
                                  shuffle=True, num_workers=args.threads,
                                  tensor_keys=["features", "labels"],
                                  tensor_dims=[encoding_dims, label_dims],
                                  tensor_neural_types=[encoding_neural_type, label_neural_type])
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=3)
    cat_acc = CategoricalAccuracy(accumulate=False)
    encoding, vz_labels = train_dataset()
    #print("train encoding.shape",encoding.shape)
    vz = model(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

    callbacks = []

    print("one step is one minibatch. one epoch is through all data.")
    print("run train, model checkpoint, eval every 32,768 data points or step=32768/my_batch_size")

    # Logger callback
    loggercallback = nemo.core.SimpleLossLoggerCallback(
        tensors=[vz_loss, vz, vz_labels],
        step_freq=32768/my_batch_size, # every 32768 data points
        print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}, Train Acc: {str(cat_acc(x[2], x[1]))}'),
    )
    callbacks.append(loggercallback)

    # Checkpointing models through NeMo callback
    checkpointcallback = nemo.core.CheckpointCallback(
        folder=args.model_dir,
        load_from_folder=None,
        # Checkpointing frequency in steps
        step_freq=32768/my_batch_size, # -1,
        # Checkpointing frequency in epochs
        #epoch_freq=512,
        # Number of checkpoints to keep
        checkpoints_to_keep= 1024,
        # If True, CheckpointCallback will raise an Error if restoring fails
        force_load=False
    )
    callbacks.append(checkpointcallback)

    # Create eval DAG if eval files are available
    if args.eval_hdf:
        eval_dataset = HDFDataLoader(args.eval_hdf, batch_size=32,
                                     shuffle=False, num_workers=args.threads,
                                     tensor_keys=["features", "labels"],
                                     tensor_dims=[encoding_dims, label_dims],
                                     tensor_neural_types=[encoding_neural_type, label_neural_type])
        eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=3)
        eval_encoding, eval_vz_labels = eval_dataset()
        #print("eval_encoding.shape",eval_encoding.shape)
        eval_vz = model(encoding=eval_encoding)
        eval_vz_loss = eval_vz_ce_loss(logits=eval_vz, labels=eval_vz_labels)

        # Add evaluation callback
        evaluator_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_vz_loss, eval_vz, eval_vz_labels],
            user_iter_callback=generate_eval_callback(CategoricalAccuracy(accumulate=True)),
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=32768/my_batch_size,
            #eval_epoch=1,
            eval_at_start=False,
        )
        callbacks.append(evaluator_callback)

    # Invoke the "train" action. /home/UNIXHOME/mbrown/.conda/envs/tf/lib/python3.8/site-packages/nemo/backends/pytorch/actions.py

    #### TODO:  can you CANNOT change optimizer after restore!!!
    # myoptimizer = nf.create_optimizer(
    #     optimizer="sgd",
    #     things_to_optimize=[vz_loss],
    #     optimizer_params={"lr": 0.01, "momentum": 0.0})
    # training_loop = [ (myoptimizer, [vz_loss]) ]
    # nf.train(training_loop,
    #          callbacks=callbacks,
    #          optimization_params={"num_epochs": args.epochs,"lr": 0.01, "momentum": 0.0})

    #myopt = torch.optim.SGD( params=[vz_loss], lr=0.01 )

    nf.train([vz_loss],
             callbacks=callbacks,
             optimization_params={"num_epochs": args.epochs, "lr": args.lr},
             optimizer=args.optimizer)


def build_parser():
    """Build parser object with options for sample."""
    parser = argparse.ArgumentParser(
        description="Read consensus trainer based on VariantWorks.")

    parser.add_argument("--local_rank", type=int,
                        help="Local rank for multi GPU training. Do not set directly.",
                        default=os.getenv('LOCAL_RANK', None))
    parser.add_argument("--train-hdf",
                        help="HDF with examples for training.",
                        required=True)
    parser.add_argument("--eval-hdf",
                        help="HDF with examples for evaluation.",
                        required=False)
    parser.add_argument("--epochs", type=int,
                        help="Epochs for training.",
                        required=False, default=1)
    import multiprocessing
    parser.add_argument("-t", "--threads", type=int,
                        help="Threads to use for parallel loading.",
                        required=False, default=multiprocessing.cpu_count())
    parser.add_argument("--model-dir", type=str,
                        help="Directory for storing trained model checkpoints. Stored after every eppoch of training.",
                        required=False, default="./models")
    parser.add_argument("--input_feature_size", type=int, default=7)
    parser.add_argument("--num_output_logits", type=int, default=25)
    parser.add_argument("--gru_size", help="Number of units in RNN", type=int, default=128)
    parser.add_argument("--gru_layers", help="Number of layers in RNN", type=int, default=2)
    parser.add_argument("--lr", help="learning rate", type=float, default=1.0E-3)
    parser.add_argument("--optimizer", help="optimizer name: adam, sgd", type=str, default="adam")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
