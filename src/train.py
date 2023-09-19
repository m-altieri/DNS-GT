import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
from colorama import Fore, Style

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from models import DELM, Word2Vec

from utils.data_loader import (
    SequenceGenerator,
    FixedSequencingStrategy,
    ClusterSequencingStrategy,
    W2VStrategy,
)
from utils.runs_management import RunManager
from utils.distribute import DummyStrategy


def config_gpus(conf):
    if isinstance(conf.get("gpu"), int):
        device = tf.config.list_physical_devices("GPU")[conf.get("gpu")]
        tf.config.set_visible_devices(device, "GPU")
        print(f"Set {device} as the only visible device.")
    for device in tf.config.get_visible_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print("Cannot enable memory growth on device:", device)
            sys.exit(e)


def get_logger(verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def build_model(model, conf, dist_strategy):
    loss_reduction = (
        tf.keras.losses.Reduction.NONE if conf.get("distribute") else "auto"
    )
    if conf.get("finetune"):
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
    with dist_strategy.scope():
        if model.lower() == "delm":
            model = DELM(conf, dist_strategy)
        elif model.lower() == "w2v":
            model = Word2Vec(conf, dist_strategy)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=conf.get("lr")),
            loss=loss,
            metrics=[],
            run_eagerly=conf.get("eager"),
        )
        return model


# TODO Refactor out argument parsing
# Also, I don't really need defaults now that I pull them from the default.yaml file
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", action="store", default="DELM")
    argparser.add_argument(
        "--es",
        action="store_true",
        help="Early Stopping",
    )
    argparser.add_argument(
        "--load",
        action="store",
        nargs="?",
        const="last",
        help="Whether to load the model from a saved checkpoint or to reinitialize a new one. Set it with no value or with 'last' to load the most recent checkpoint, or manually set a checkpoint file name.",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug information",
    )
    argparser.add_argument(
        "--epochs",
        action="store",
        type=int,
        help="Number of training epochs",
    )
    argparser.add_argument(
        "--bs", action="store", default=512, type=int, help="Batch size"
    )
    argparser.add_argument(
        "--lr", action="store", default=1e-4, type=float, help="Learning rate"
    )
    argparser.add_argument(
        "--demo",
        action="store_true",
        help="Used for debugging purposes",
    )
    argparser.add_argument(
        "--test-seq",
        action="store",
        type=int,
        help="Used for debugging purposes; choose the test sequence index",
    )
    argparser.add_argument(
        "--seqlen",
        action="store",
        type=int,
        help="Maximum sequence length",
    )
    argparser.add_argument(
        "--stride",
        action="store",
        type=int,
        help="Stride between sequences (how many queries to shift by)",
    )
    argparser.add_argument(
        "--include-start",
        action="store_true",
        help="Whether to include <START> as the first token of each sequence (total length is unaffected)",
    )
    argparser.add_argument(
        "--version",
        action="store",
        choices=[
            "small",
            "all",
            "clean",
        ],  # TODO clean should become the normal (and only) one
        default="clean",
        help="Deprecated. Version of the dataset used.",  # TODO deprecate
    )
    argparser.add_argument(
        "--tiny",
        action="store_true",
        help="Use for debugging purposes, to use a tiny portion of the dataset to get faster feedback.",
    )
    argparser.add_argument(
        "--gpu",
        action="store",
        help="If it is an integer (eg. --gpu 3), run on a single specific GPU. "
        + "If it is an array (eg. [2,4]), distribute the execution on the specified GPUs. "  # TODO
        + "If it is `all`, distribute on all GPUs.",
    )
    argparser.add_argument("--tensorboard", "--tb", action="store_true")
    argparser.add_argument(
        "--quick-tb",
        action="store_true",
        help="Whether to reutilize the same TensorBoard folder. Allows for quicker debugging.",
    )
    argparser.add_argument("--eager", action="store_true")
    argparser.add_argument("--blocks", action="store", type=int)
    argparser.add_argument(
        "--group-by-host", action="store", default=True, type=bool
    )
    argparser.add_argument(
        "--run-name",
        action="store",
        default=f'model-{time.strftime("%y%m%d-%H%M%S", time.localtime())}',
        help="Name used when saving to file. Has no effect if --load.",
    )
    argparser.add_argument("--omega", action="store", type=float)
    argparser.add_argument("--shuffle", action="store_true")  # deprecated
    argparser.add_argument(
        "--type",
        action="store",
        help="Model type. It is used by model classes that have multiple subtypes, like Word2Vec.",
    )
    argparser.add_argument(
        "--dim",
        action="store",
        type=int,
        help="Dimension of the embeddings. Now host and domain embeddings always have the same dimension.",
    )
    argparser.add_argument(
        "--finetune",
        "--ft",
        action="store_true",
        help="Use the classification workflow instead of the embedding learning workflow.",
    )
    argparser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze all layers except the classification ones. Only has effect if --finetune.",
    )
    argparser.add_argument(
        "--test-partition",
        type=int,
        help="Test partition to choose during finetuning. "
        + "Only has effect if --finetune.",
    )
    argparser.add_argument(
        "--test-fold",
        type=int,
        help="Test fold to choose during finetuning. "
        + "Domains contained in the fold will not be used for loss computation, making them suitable for testing. "
        + "If not set, all domains will be considered during loss computation. "
        + "Only has effect if --finetune.",
    )
    argparser.add_argument(
        "--from-pretrained",
        "--from-pt",
        action="store_true",
        help="Whether to load weights from existing finetuned model or from pretrained model. "
        + "Only has effect if --finetune.",
    )
    argparser.add_argument("--max-tokens", action="store", type=int)
    argparser.add_argument("--concat-hosts", action="store_true")
    argparser.add_argument(
        "--seq-strategy",
        action="store",
        choices=["cluster", "fixed"],
    )
    argparser.add_argument("--evaluate", action="store_true")

    args = argparser.parse_args()

    assert args.test_seq is None or args.test_seq > 0

    try:
        args.gpu = int(args.gpu)
    except:  # it is not a number, it's either None or `all`
        pass
    try:
        if "[" in args.gpu:  # if it is a list
            args.gpu = [
                int(i) for i in args.gpu.strip("[").strip("]").split(",")
            ]
    except:  # it is not a list, let's try with a number
        pass
    if isinstance(args.gpu, list) or args.gpu == "all":
        args.distribute = True
        assert tf.config.get_visible_devices(
            "GPU"
        ) == tf.config.list_physical_devices(
            "GPU"
        )  # if distribute, devices cannot be set as not visible, to avoid possible bugs
    else:
        args.distribute = False

    return args


def find_last_checkpoint(dir):
    if len(os.listdir(dir)) > 0:
        checkpoint = os.listdir(dir)[
            [
                os.path.getmtime(os.path.join(dir, f)) for f in os.listdir(dir)
            ].index(
                max(
                    [
                        os.path.getmtime(os.path.join(dir, f))
                        for f in os.listdir(dir)
                    ]
                )
            )
        ]
    else:
        checkpoint = ""
    return checkpoint


def default_checkpoint(conf):
    return f"{conf.get('run_name')}.h5"


def indent(depth=1):
    return f"".join(["--" for i in range(depth - 1)]) + "> "


def main():
    args = parse_args()

    # Initialize logger
    logger = get_logger(args.verbose)
    logger.info(f"Starting program with args: {vars(args)}\n")

    # Manage save folder
    run_manager = RunManager(
        model_object=None,
        model_name=f"{args.model}{f'-{args.type}' if args.type else ''}",
        run_name=args.run_name,
        last=args.load == "last",
        verbose=True,
    )

    # Manage model configuration
    conf = run_manager.load_conf()

    superseding_args = {
        k: v
        for (k, v) in vars(args).items()
        if conf.get(k) != v and v is not None
    }

    conf = conf | superseding_args
    conf_log = "\n".join(
        [
            f"{Fore.YELLOW if k in superseding_args and conf != superseding_args else ''}{indent(1)}{k:<20}: {v}{Fore.RESET}"
            for (k, v) in conf.items()
        ]
    )
    logger.info(f"Configuration: \n{conf_log}")

    config_gpus(conf)

    sequencing_strategy = None
    if conf.get("seq_strategy") == "fixed":
        sequencing_strategy = FixedSequencingStrategy()
    elif conf.get("seq_strategy") == "cluster":
        sequencing_strategy = ClusterSequencingStrategy()
    elif conf.get("model") == "W2V":
        sequencing_strategy = W2VStrategy()
    else:
        raise ValueError()

    train = tf.data.Dataset.from_generator(
        SequenceGenerator(
            os.path.join(conf.get("queries_path"), "train"),
            sequencing_strategy,
            conf.get("seqlen"),
            conf.get("finetune"),
            conf.get("group_by_host"),
            stride=conf.get("stride"),
            include_start=conf.get("include_start"),
        ),
        output_signature=tf.TensorSpec(
            shape=[conf.get("seqlen"), 2 + conf.get("finetune")],
            dtype=tf.string,
        )
        if conf.get("model").lower() == "delm"
        else tf.TensorSpec(
            shape=[conf.get("seqlen"), 1 + conf.get("finetune")],
            dtype=tf.string,
        ),
    )
    test = tf.data.Dataset.from_generator(
        SequenceGenerator(
            os.path.join(conf.get("queries_path"), "test"),
            sequencing_strategy,
            conf.get("seqlen"),
            conf.get("finetune"),
            conf.get("group_by_host"),
            stride=conf.get("stride"),
            include_start=conf.get("include_start"),
        ),
        output_signature=tf.TensorSpec(
            shape=[conf.get("seqlen"), 2 + conf.get("finetune")],
            dtype=tf.string,
        )
        if conf.get("model").lower() == "delm"
        else tf.TensorSpec(
            shape=[conf.get("seqlen"), 1 + conf.get("finetune")],
            dtype=tf.string,
        ),
    )
    if not conf.get("demo") and conf.get("shuffle"):
        train = train.shuffle(1000000)
    train = train.batch(conf.get("bs")).prefetch(tf.data.AUTOTUNE)
    test = test.batch(conf.get("bs")).prefetch(tf.data.AUTOTUNE)

    # Manage training distribution
    dist_strategy = None
    if conf.get("distribute"):
        gpus = (
            [f"/gpu:{i}" for i in conf.get("gpu")]
            if isinstance(conf.get("gpu"), list)
            else None
        )  # initializing MirroredStrategy with None uses all gpus
        dist_strategy = tf.distribute.MirroredStrategy(gpus)
        logger.warning(
            f"{Fore.BLUE}Distributing on {dist_strategy.num_replicas_in_sync} devices.{Style.RESET_ALL}"
        )

        # Data Config
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        train = train.with_options(options)
        test = test.with_options(options)
        train = dist_strategy.experimental_distribute_dataset(train)
        test = dist_strategy.experimental_distribute_dataset(test)
    else:
        dist_strategy = DummyStrategy

    # Build Model
    model = build_model(conf.get("model"), conf, dist_strategy=dist_strategy)

    # Save the updated configuration
    run_manager.model_object = model
    run_manager.save_conf()

    # If finetune, freeze all layers but the last classification layer,
    # otherwise, unfreeze in case it was frozen
    if conf.get("finetune"):
        model.finetune()
    else:
        model.pretrain()

    logger.info(f"Calling model to initialize layers...")
    if conf.get("distribute"):
        model.distributed_test_step(next(iter(test)))
    else:
        model.test_step(next(iter(test)))

    if run_manager.exist_weights():
        model = run_manager.load_weights(model)

    # TODO demo should be refactored out or cleaned. it's too much in the way now
    if conf.get("demo"):
        conf["eager"] = True
        conf["tensorboard"] = True
        conf["gpu"] = None
        conf["distribute"] = False
        conf["bs"] = 1

        logger.info(
            f"{Style.BRIGHT}\nDomain Embeddings Language Model{Style.RESET_ALL}\n"
            + "Please refer to https://gitlab.jrc.ec.europa.eu/jrc-projects/createg/cdp-bari/dns/-/tree/main/ for roadmap and updates.\n"
            + "Syntax: <Host> <Domain> -> <Predicted Domain> (<prob%>) [(<Unmasked Domain> <prob%>)]\n"
        )
        seq_idx = conf.get("test_seq") or np.random.randint(0, 1000)
        seqs = (
            test.unbatch()
            .skip(seq_idx)
            .shuffle(1000)
            .take(5)
            .as_numpy_iterator()
        )
        seqs = np.array([s for s in seqs], dtype=object)
        for s in range(len(seqs)):
            seq = seqs[s : s + 1]

            mask = np.zeros_like(seq)
            # place 1's where you want to replace tokens with <MASK>
            # axis 0 is always 0 (array of length 1), axis 1 is the index of token within the sequence, axis 2 is 0 for host and 1 for domain
            # example: mask[0, 1, 1]
            #   always zero ^  ^  ^
            #     second token |  |
            #                     | domain
            mask[0, 0, -1] = 1
            # mask[0, 1, -1] = 1
            # mask[0, 2, -1] = 1
            masked_seq = np.where(
                mask, np.full_like(seq, b"<MASK>", dtype=object), seq
            )

            pred, loss, kwout = model._predict(seq, mask)
            print(f"Seq index: {seq_idx + s}")

            if conf.get("finetune"):
                pred = np.array(pred).flatten()
                for p, _ in enumerate(pred):
                    domain = seq[0, p, 1]

                    label = seq[0, p, 2]

                    print(
                        f"{Fore.CYAN if kwout.get('in_fold')[p] else ''}{domain} ({label}) -> {pred[p]:.3f}{Style.RESET_ALL}"
                    )
                print(f"{Style.BRIGHT}Loss: {loss:.3f}{Style.RESET_ALL}")
            else:
                pred = pred[0]

                for i in range(len(pred)):
                    # I am actually interested in what token the model considers, not what we pass as input (if the token is not in the vocabulary, it will be treated as <UNK>)
                    masked_host = model.inverse_hosts_lookup(
                        model.hosts_lookup(masked_seq[0, i, 0])
                    )

                    domain = model.inverse_domains_lookup(
                        model.domains_lookup(seq[0, i, 1])
                    )

                    masked_domain = model.inverse_domains_lookup(
                        model.domains_lookup(masked_seq[0, i, 1])
                    )

                    predicted_token = model.inverse_domains_lookup(
                        np.array(pred).argmax(axis=-1)[i]
                    )
                    domain_index = model.domains_lookup(domain)
                    logger.info(
                        f"{masked_host} {masked_domain} -> {f'{Fore.GREEN}' if domain == predicted_token else f'{Fore.RED}'}{predicted_token} ({100*(np.array(pred).max(axis=-1)[i]):.2f}%){Style.RESET_ALL} {f'{Style.DIM}({domain} {100*(np.array(pred)[i,domain_index]):.2f}%) {Style.RESET_ALL}' if not domain == predicted_token else ''}"
                    )
                logger.info(f"{Style.BRIGHT}Loss: {loss:.3f}{Style.RESET_ALL}")

        sys.exit(0)

    if conf.get("verbose"):
        logger.info(model.summary())

    logger.info("Starting model training...")

    # <------- EXPERIMENTAL: reuniting distributed and non-distributed training loops + externalizing saving
    train_steps_per_epoch = None
    test_steps_per_epoch = None
    for epoch in range(conf.get("epochs")):
        pbar = tqdm(train, total=train_steps_per_epoch)

        # Training loop
        step = 0
        total_loss = 0.0
        for x in pbar:
            step += 1
            total_loss += (
                model.train_step(x)
                if not conf.get("distribute")
                else model.distributed_train_step(x)
            )
            pbar.set_description(
                f"[Epoch {epoch+1}/{conf.get('epochs')}] Train Loss: {total_loss / step:.4f}"
            )
        train_steps_per_epoch = step

        # save weights
        run_manager.save_weights(model)

        # Test loop
        step = 0
        total_loss = 0.0
        pbar = tqdm(test, total=test_steps_per_epoch)
        for x in pbar:
            step += 1
            total_loss += (
                model.test_step(x)
                if not conf.get("distribute")
                else model.distributed_test_step(x)
            )
            pbar.set_description(
                f"[Epoch {epoch+1}/{conf.get('epochs')}] Test Loss: {total_loss / step:.4f}"
            )
        test_steps_per_epoch = step

    # Save embeddings
    logger.info("Saving model embeddings...")
    run_manager.save_embeddings(model)
    # --------------------->

    # Model Evaluation TODO refactor out
    if conf.get("evaluate"):
        logger.info("Starting model evaluation...")

        # TODO implement saving model predictions in distributed mode
        if conf.get("distribute"):
            raise NotImplementedError(
                f"{Fore.YELLOW}Saving model predictions is not supported in --distribute.{Style.RESET_ALL}"
            )

        # ISSUE Decide if I should evaluate on train, test or both. Consider that:
        # (1) in any case the model is never trained on in_fold domains, even on train
        # (2) it may be easier to predict on sequences that the model has already seen
        # NOTE I'm evaluating on test

        # <--- Saving Predictions
        d, trues, preds = [], [], []
        step = 0
        for x in tqdm(test):
            # DELM: [B,L,3] (host,domain,label), W2V-CBOW: [B,L,2] (domain,label)

            # Compute predictions on x
            domains_dim = 1 if conf.get("model") == "DELM" else 0
            domains = x[..., domains_dim]
            true = x[..., -1]
            pred, _, in_fold_mask = model._predict(x)

            # Only take predictions for domains that are in the test fold
            # I want all these to be [B,]
            domains = domains[in_fold_mask]
            true = true[in_fold_mask]
            pred = pred[in_fold_mask]

            # Convert from tf.Tensors to lists
            domains = domains.numpy()
            domains = [domain.decode("utf-8") for domain in domains]
            true = [int(y) for y in true]

            # Append current predictions in a flattened way
            d.extend(np.ravel(domains))
            trues.extend(np.ravel(true))
            preds.extend(np.ravel(pred))

            step += 1

        # Create predictions DataFrame
        df = pd.DataFrame({"domains": d, "true": trues, "pred": preds})
        print(df)

        # Save predictions DataFrame
        run_manager.save_predictions(
            df, conf.get("test_partition"), conf.get("test_fold")
        )
        # --->

        # <--- Compute metrics. TODO Refactor out and save them through RunManager

        # Confusion matrix
        logger.info(
            f"SKlearn Confusion matrix:\n{sklearn.metrics.confusion_matrix(df['true'], df['pred'].round(), labels=[1,0])}"
        )

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            preds_hard = (df["pred"] > threshold).astype(int)
            logger.info(
                f"Precision: {sklearn.metrics.precision_score(df['true'],preds_hard):.2f}"
            )
            logger.info(
                f"Recall: {sklearn.metrics.recall_score(df['true'], preds_hard):.2f}"
            )
            logger.info(
                f"F1 score: {sklearn.metrics.f1_score(df['true'], preds_hard):.2f}"
            )
            logger.info(
                f"Accuracy: {sklearn.metrics.accuracy_score(df['true'], preds_hard):.2f}"
            )

        # AUC
        auc = sklearn.metrics.roc_auc_score(df["true"], df["pred"])
        logger.info(f"AUC: {auc}")

        # Plot ROC
        fpr, tpr, _ = sklearn.metrics.roc_curve(df["true"], df["pred"])
        plt.figure(figsize=(10, 10))
        plt.plot(
            fpr,
            tpr,
            label=f"ROC fold {conf.get('test_fold')} (AUC = {auc:.2f})",
        )
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(
                run_manager.run_path,
                f"""roc-{conf.get("model")}{f"-{conf.get('type')}" if conf.get('type') else ""}{f"-{conf.get('test_partition')}-{conf.get('test_fold')}"}.png""",
            )
        )
        # --->


if __name__ == "__main__":
    main()
